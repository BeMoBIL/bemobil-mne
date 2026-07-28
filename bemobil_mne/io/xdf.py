"""Multimodal XDF loader for MoBI and physiological recordings.

Handles streams at different sampling rates (EEG, ECG, EMG, eye-tracking,
EDA, audio, video) by separating them into two tiers and aligning Tier-1
streams to a common grid via timestamp-aware interpolation.

See bemobil_mne/io/README.md for full design rationale.
"""

# %% Imports

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mne
import numpy as np
import pyxdf

from bemobil_mne.io.alignment import (
    _compute_effective_srate,
    _is_irregular,
    align_stream_to_timestamps,
)

# %% Settings & Constants

logger = logging.getLogger(__name__)

_MICROVOLT_UNITS = frozenset(("microvolt", "microvolts", "µv", "μv", "uv"))
_VOLT_UNITS = frozenset(("v", "volt", "volts"))

# %% Data container


@dataclass
class MultimodalRecording:
    """Container returned by :meth:`XDFLoader.load`.

    Attributes
    ----------
    raw : mne.io.BaseRaw
        Tier-1 physiological streams (EEG, ECG, EMG, eye-tracking, EDA, ...)
        aligned to a common grid at ``target_sfreq`` and merged into an MNE
        Raw object.  The time axis starts at 0 and corresponds to LSL
        timestamp ``session_t0``.
    tier2 : dict[str, tuple[np.ndarray, np.ndarray]]
        Tier-2 streams kept at their native sampling rate, suitable for
        time-frequency analysis or frame-level processing.
        ``{label: (data, timestamps_s)}`` where *data* is
        ``(n_samples, n_channels)`` and *timestamps_s* are **seconds relative
        to session_t0**.
    events : dict[str, tuple[list[str], np.ndarray]]
        Irregular / event-based streams (``nominal_srate == 0``), e.g.
        fixation events, saccade events, or custom trigger streams.
        ``{label: (descriptions, timestamps_s)}`` relative to ``session_t0``.
    session_t0 : float
        Absolute LSL timestamp of t = 0 in the Raw object (and in all Tier-2
        and event timestamp arrays).  Use this to cross-reference with external
        system logs that carry LSL time.
    """

    raw: mne.io.BaseRaw
    tier2: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    events: dict[str, tuple[list[str], np.ndarray]] = field(default_factory=dict)
    session_t0: float = 0.0


# %% XDF-specific helpers (MNE-dependent; stay in this module)


def _unit_scale(unit: str, ch_name: str = "") -> float:
    """Return scale factor to convert *unit* to SI volts (case-insensitive).

    Emits a warning for unrecognised units so silent identity scaling is visible.
    """
    normed = unit.strip().lower()
    if normed in _MICROVOLT_UNITS:
        return 1e-6
    if normed not in _VOLT_UNITS and normed not in ("na", ""):
        logger.warning(
            "Unrecognised unit '%s' for channel '%s' -- assuming volts "
            "(no scaling applied).",
            unit,
            ch_name,
        )
    return 1.0


def _match_stream(
    streams: list[dict],
    name: str | None = None,
    stream_type: str | None = None,
    source_id: str | None = None,
    allow_multiple: bool = False,
) -> dict | list[dict] | None:
    """Match streams by name, type, and/or source_id."""
    matched = []
    for s in streams:
        info = s["info"]
        if name and info.get("name", [""])[0] != name:
            continue
        if stream_type and info.get("type", [""])[0].lower() != stream_type.lower():
            continue
        if source_id and info.get("source_id", [""])[0] != source_id:
            continue
        matched.append(s)

    if not matched:
        return [] if allow_multiple else None
    if allow_multiple:
        return matched
    if len(matched) > 1:
        logger.warning(
            "Multiple streams matched (name=%s, type=%s). Using first.",
            name,
            stream_type,
        )
    return matched[0]


def _extract_channel_info(stream: dict) -> tuple[list[str], list[str], list[str]]:
    """Extract channel labels, types, and units from a stream."""
    n_chans = int(stream["info"]["channel_count"][0])
    labels, types, units = [], [], []
    try:
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            ch_type = ch.get("type", ["misc"])[0].lower()
            types.append(
                ch_type if ch_type in mne.io.get_channel_type_constants() else "misc"
            )
            units.append(ch.get("unit", ["NA"])[0])
    except (TypeError, IndexError, KeyError):
        pass

    if not labels:
        labels = [f"ch_{i}" for i in range(n_chans)]
    if not types:
        types = ["misc"] * n_chans
    if not units:
        units = ["NA"] * n_chans

    labels = _shorten_fif_ch_names(labels)
    return labels, types, units


# FIF format caps channel names at 15 characters.  This lookup table maps
# known long names from Pupil Labs eye-tracking LSL streams to MNE-style
# names that fit within the limit.
#
# MNE convention (from mne.io.eyelink): lowercase, ``_left`` / ``_right``
# suffix for eye identity, ``_x`` / ``_y`` / ``_z`` for axis.  Channel types
# ``"eyegaze"`` (gaze position) and ``"pupil"`` (pupil size) are MNE built-ins.
# For 3D eyeball / optical-axis / eyelid channels that have no MNE built-in
# we extend the same pattern: ``<metric>_<axis>_<eye>``.
#
# Pupil Labs indices: 0 = left eye, 1 = right eye.
_FIF_CH_ABBREVIATIONS: dict[str, str] = {
    # ---- Pupil diameter (MNE channel type: pupil) -----------------------
    "PupilDiameter-0": "pupil_left",
    "PupilDiameter-1": "pupil_right",
    # ---- 3-D eyeball centre (MNE channel type: misc) --------------------
    "EyeballCenterX-0": "eyeball_x_l",
    "EyeballCenterY-0": "eyeball_y_l",
    "EyeballCenterZ-0": "eyeball_z_l",
    "EyeballCenterX-1": "eyeball_x_r",
    "EyeballCenterY-1": "eyeball_y_r",
    "EyeballCenterZ-1": "eyeball_z_r",
    # ---- Optical axis (MNE channel type: misc) --------------------------
    "OpticalAxisX-0": "optaxis_x_l",
    "OpticalAxisY-0": "optaxis_y_l",
    "OpticalAxisZ-0": "optaxis_z_l",
    "OpticalAxisX-1": "optaxis_x_r",
    "OpticalAxisY-1": "optaxis_y_r",
    "OpticalAxisZ-1": "optaxis_z_r",
    # ---- Eyelid geometry (MNE channel type: misc) -----------------------
    # Suffix: t = top lid, b = bottom lid; l = left eye, r = right eye
    "EyelidAngleTopLeft": "eyelid_ang_tl",
    "EyelidAngleBottomLeft": "eyelid_ang_bl",
    "EyelidApertureLeft": "eyelid_apt_l",
    "EyelidAngleTopRight": "eyelid_ang_tr",
    "EyelidAngleBottomRight": "eyelid_ang_br",
    "EyelidApertureRight": "eyelid_apt_r",
}

_FIF_MAX_CH_LEN = 15


def _shorten_fif_ch_names(
    labels: list[str], max_len: int = _FIF_MAX_CH_LEN
) -> list[str]:
    """Shorten channel names that exceed *max_len* characters.

    Uses :data:`_FIF_CH_ABBREVIATIONS` for known eye-tracker channels and
    falls back to deterministic truncation-with-deduplication for any other
    name.  Logs all renames so the mapping remains auditable.

    Parameters
    ----------
    labels : list of str
        Original channel names from the XDF stream descriptor.
    max_len : int
        Maximum allowed name length (FIF limit = 15).

    Returns
    -------
    list of str
        Names guaranteed to be ``<= max_len`` characters, unique within the
        returned list.
    """
    seen: set[str] = set()
    out: list[str] = []

    for name in labels:
        if len(name) <= max_len:
            out.append(name)
            seen.add(name)
            continue

        # 1. Try the lookup table first
        short = _FIF_CH_ABBREVIATIONS.get(name)

        # 2. Generic fallback: truncate and disambiguate with a counter
        if short is None or len(short) > max_len:
            short = name[:max_len]

        # Disambiguate if the shortened name already appeared
        if short in seen:
            suffix_len = 3
            base = short[: max_len - suffix_len]
            for i in range(1, 1000):
                candidate = f"{base}{i:0{suffix_len - 1}d}"
                if candidate not in seen:
                    short = candidate
                    break

        logger.info(
            "XDF channel name %r (%d chars) shortened to %r for FIF compatibility.",
            name,
            len(name),
            short,
        )
        out.append(short)
        seen.add(short)

    return out


def _crop_stream(
    stream: dict, start_time: float, end_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crop stream to [start_time, end_time]."""
    ts = np.array(stream["time_stamps"])
    data = np.array(stream["time_series"])
    mask = (ts >= start_time) & (ts <= end_time)
    return data[mask], ts[mask]


# %% Main class


class XDFLoader:
    """Multimodal XDF loader for MoBI and physiological recordings.

    Streams are split into two tiers:

    **Tier 1** -- continuous physiological streams merged into an MNE Raw
    object at a common ``target_sfreq``.  Includes EEG (primary), plus any
    streams listed in ``special_streams`` (ECG, EMG, eye-tracking, EDA, ...).

    **Tier 2** -- high-rate or frame-based streams kept at their native
    sampling rate and returned separately as
    ``{label: (data, timestamps_s)}``.  Intended for audio (requiring
    time-frequency analysis at full bandwidth) and video (raw frames).

    Irregular streams (``nominal_srate == 0``) -- e.g. fixation events,
    saccade events, custom triggers -- are routed to the ``events`` dict
    rather than either tier.

    Parameters
    ----------
    eeg_stream_name : str, optional
        Name of the EEG stream to load.
    eeg_stream_type : str
        LSL type of the EEG stream.  Default: ``"EEG"``.
    eeg_source_id : str, optional
        Source ID of the EEG stream.
    marker_stream_types : list[str]
        LSL types to scan for marker / annotation streams.
        Default: ``["Markers", "Logging", "Notes"]``.
    special_streams : dict, optional
        Tier-1 auxiliary streams to align and merge into the Raw object.
        Format: ``{label: {"name": <str>, "type": <str>, "method": <str>}}``.
        The optional ``"method"`` key overrides ``alignment_method`` for that
        stream only, allowing per-stream control (e.g. ``"sinc"`` for ECG at
        1000 Hz while EDA uses ``"pchip"`` and a button box uses ``"stim"``).
        Examples: ECG amplifier, eye-tracking gaze, EMG, EDA device.
    tier2_streams : dict, optional
        Tier-2 streams kept at native rate (not merged into Raw).
        Same format as ``special_streams`` (``"method"`` key is ignored here).
        Examples: audio at 44.1 kHz, video stream.
    montage : str, Path, or DigMontage, optional
        Montage applied to EEG channels.
    old_reference : str, optional
        Reference channel to restore before setting montage (e.g. ``"FCz"``).
    keep_channels : list[str], optional
        Channel names or types to keep; all others are dropped.
    drop_channels : list[str], optional
        Channel names or types to drop.
    target_sfreq : float, optional
        Common sampling frequency for all Tier-1 streams.  EEG is resampled
        to this rate via MNE's anti-aliased resample; auxiliary streams are
        aligned to the resulting time grid.  Defaults to the EEG stream's
        native rate if not set.
    alignment_method : {'linear', 'pchip', 'sinc', 'nearest', 'stim'}
        Default interpolation method for all Tier-1 auxiliary streams.
        Can be overridden per stream via the ``"method"`` key in
        ``special_streams``.  Default: ``'pchip'``.
    max_nan_gap_s : float or None
        NaN gaps longer than this (seconds) in auxiliary streams are preserved
        in the output rather than bridged.  ``None`` fills all gaps.
    on_mismatch : {'crop', 'pad'}
        Behaviour when auxiliary streams do not fully cover the EEG time range.

        ``'crop'`` -- trim the recording to the intersection of all Tier-1
        stream time ranges.  Safe when losing a few edge seconds is acceptable.

        ``'pad'`` -- keep the full EEG time range; auxiliary streams that do
        not cover the edges receive ``np.nan`` there (zeroed in Raw, marked
        with ``BAD_<label>_missing`` annotations).  Preferable for continuous
        recordings where trimming would discard usable data.
    """

    def __init__(
        self,
        eeg_stream_name: str | None = None,
        eeg_stream_type: str = "EEG",
        eeg_source_id: str | None = None,
        marker_stream_types: list[str] | None = None,
        special_streams: dict[str, dict] | None = None,
        tier2_streams: dict[str, dict] | None = None,
        montage: str | mne.channels.DigMontage | None = None,
        old_reference: str | None = None,
        keep_channels: list[str] | None = None,
        drop_channels: list[str] | None = None,
        target_sfreq: float | None = None,
        alignment_method: Literal[
            "linear", "pchip", "sinc", "nearest", "stim"
        ] = "pchip",
        max_nan_gap_s: float | None = None,
        on_mismatch: Literal["crop", "pad"] = "crop",
    ):
        self.eeg_stream_name = eeg_stream_name
        self.eeg_stream_type = eeg_stream_type
        self.eeg_source_id = eeg_source_id
        self.marker_stream_types = marker_stream_types or [
            "Markers",
            "Logging",
            "Notes",
        ]
        self.special_streams = special_streams or {}
        self.tier2_streams = tier2_streams or {}
        self.montage = montage
        self.old_reference = old_reference
        self.keep_channels = keep_channels
        self.drop_channels = drop_channels
        self.target_sfreq = target_sfreq
        self.alignment_method = alignment_method
        self.max_nan_gap_s = max_nan_gap_s
        self.on_mismatch = on_mismatch

    def load(self, path: str | Path) -> MultimodalRecording:
        """Load an XDF file and return a :class:`MultimodalRecording`.

        Parameters
        ----------
        path : str or Path
            Path to the XDF file.

        Returns
        -------
        MultimodalRecording
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"XDF file not found: {path}")
        if path.suffix.lower() != ".xdf":
            raise ValueError(f"Expected an XDF file, got: {path.suffix}")

        streams, _ = pyxdf.load_xdf(str(path))
        logger.info("Loaded %d streams from %s", len(streams), path.name)
        for s in streams:
            info = s["info"]
            logger.info(
                "  Stream: name='%s' type='%s' source_id='%s'",
                info.get("name", [""])[0],
                info.get("type", [""])[0],
                info.get("source_id", [""])[0],
            )

        raw, tier2, events, session_t0 = self._build_raw(streams)

        if self.old_reference:
            raw.add_reference_channels(self.old_reference)

        if self.montage:
            self._set_montage(raw)

        if self.target_sfreq:
            raw.resample(self.target_sfreq)

        self._apply_channel_selection(raw)

        return MultimodalRecording(
            raw=raw,
            tier2=tier2,
            events=events,
            session_t0=session_t0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_raw(
        self, streams: list[dict]
    ) -> tuple[
        mne.io.BaseRaw,
        dict[str, tuple[np.ndarray, np.ndarray]],
        dict[str, tuple[list[str], np.ndarray]],
        float,
    ]:
        """Build MNE Raw, Tier-2 dict, and events dict from XDF streams."""
        # -- EEG stream --------------------------------------------------
        eeg_stream = _match_stream(
            streams,
            name=self.eeg_stream_name,
            stream_type=self.eeg_stream_type,
            source_id=self.eeg_source_id,
        )
        if eeg_stream is None:
            raise RuntimeError(
                f"Could not find EEG stream (name={self.eeg_stream_name}, "
                f"type={self.eeg_stream_type}, source_id={self.eeg_source_id})"
            )

        labels, types, units = _extract_channel_info(eeg_stream)
        eeg_native_sfreq = float(np.array(eeg_stream["info"]["effective_srate"]).item())
        scale = np.array([_unit_scale(u, ch) for u, ch in zip(units, labels)])

        eeg_t_start = float(eeg_stream["time_stamps"][0])
        eeg_t_end = float(eeg_stream["time_stamps"][-1])

        # -- Locate Tier-1 auxiliary streams -----------------------------
        special_stream_objs: dict[str, dict] = {}
        for label, spec in self.special_streams.items():
            s = _match_stream(
                streams, name=spec.get("name"), stream_type=spec.get("type")
            )
            if s is None:
                logger.warning("Special stream '%s' not found -- skipping.", label)
                continue
            if _is_irregular(s):
                logger.warning(
                    "Special stream '%s' has nominal_srate=0 and will be "
                    "routed to events, not merged into Raw.",
                    label,
                )
                continue
            if len(s["time_stamps"]) < 2:
                logger.warning(
                    "Special stream '%s' has fewer than 2 samples -- skipping.", label
                )
                continue
            special_stream_objs[label] = s

        # -- Determine common time window ---------------------------------
        if self.on_mismatch == "crop":
            start_time = eeg_t_start
            end_time = eeg_t_end
            for s in special_stream_objs.values():
                start_time = max(start_time, float(s["time_stamps"][0]))
                end_time = min(end_time, float(s["time_stamps"][-1]))
        else:  # pad -- keep full EEG range; aux streams fill with NaN outside range
            start_time = eeg_t_start
            end_time = eeg_t_end

        session_t0 = start_time

        # -- Build EEG RawArray -------------------------------------------
        eeg_data, _ = _crop_stream(eeg_stream, start_time, end_time)
        info = mne.create_info(ch_names=labels, sfreq=eeg_native_sfreq, ch_types=types)
        raw = mne.io.RawArray((eeg_data * scale).T, info, verbose=False)

        # -- Align and append Tier-1 auxiliary streams --------------------
        # Target timestamps span the EEG window at native EEG rate.
        # If target_sfreq is set, MNE's resample runs after _build_raw and
        # resamples all channels (EEG + aux) together with its own AA filter.
        n_eeg_samples = eeg_data.shape[0]
        tgt_ts = np.linspace(start_time, end_time, n_eeg_samples)

        for label, s in special_stream_objs.items():
            src_data = np.array(s["time_series"], dtype=float)
            src_ts = np.array(s["time_stamps"])

            # Per-stream method overrides the global default
            stream_method = self.special_streams[label].get(
                "method", self.alignment_method
            )

            aligned = align_stream_to_timestamps(
                data=src_data,
                src_ts=src_ts,
                tgt_ts=tgt_ts,
                method=stream_method,
                fill_value=np.nan,
                max_nan_gap_s=self.max_nan_gap_s,
            )

            # MNE RawArray does not accept NaN -- zero-fill and annotate gaps
            nan_mask = np.isnan(aligned).any(axis=1)
            if nan_mask.any():
                aligned[nan_mask] = 0.0
                self._annotate_nan_regions(raw, nan_mask, tgt_ts, label, session_t0)

            ch_labels, ch_types, _ = _extract_channel_info(s)
            aux_info = mne.create_info(
                ch_names=ch_labels, sfreq=eeg_native_sfreq, ch_types=ch_types
            )
            aux_raw = mne.io.RawArray(aligned.T, aux_info, verbose=False)
            try:
                raw.add_channels([aux_raw], force_update_info=True)
            except ValueError:
                aux_raw.rename_channels(
                    {ch: f"{label}_{ch}" for ch in aux_raw.ch_names}
                )
                raw.add_channels([aux_raw], force_update_info=True)

            logger.info(
                "Added Tier-1 stream '%s' (%d channels, method=%s)",
                label,
                len(ch_labels),
                stream_method,
            )

        # -- Collect Tier-2 streams (native rate, timestamps relative to t0) -
        tier2: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, spec in self.tier2_streams.items():
            s = _match_stream(
                streams, name=spec.get("name"), stream_type=spec.get("type")
            )
            if s is None:
                logger.warning("Tier-2 stream '%s' not found -- skipping.", label)
                continue
            t2_data = np.array(s["time_series"], dtype=float)
            t2_ts = np.array(s["time_stamps"]) - session_t0
            tier2[label] = (t2_data, t2_ts)
            logger.info(
                "Stored Tier-2 stream '%s' at native rate (~%.1f Hz, %d samples)",
                label,
                _compute_effective_srate(np.array(s["time_stamps"])),
                len(t2_ts),
            )

        # -- Marker streams -> MNE annotations ---------------------------
        events: dict[str, tuple[list[str], np.ndarray]] = {}

        for stream_type in self.marker_stream_types:
            matched = _match_stream(
                streams, stream_type=stream_type, allow_multiple=True
            )
            for s in matched:
                onsets = np.array(s["time_stamps"]) - session_t0
                descriptions = [item for sub in s["time_series"] for item in sub]
                raw.annotations.append(onsets, [0.0] * len(onsets), descriptions)
                logger.info(
                    "Added %d markers from stream '%s'",
                    len(descriptions),
                    s["info"].get("name", ["?"])[0],
                )

        # -- Remaining irregular streams -> events dict -------------------
        handled_names = {eeg_stream["info"].get("name", [""])[0]}
        for label in list(self.special_streams) + list(self.tier2_streams):
            spec = (
                self.special_streams.get(label) or self.tier2_streams.get(label)
            ) or {}
            if spec.get("name"):
                handled_names.add(spec["name"])

        for s in streams:
            stream_name = s["info"].get("name", [""])[0]
            stream_type_val = s["info"].get("type", [""])[0]
            if stream_name in handled_names:
                continue
            if stream_type_val in self.marker_stream_types:
                continue
            if _is_irregular(s) and len(s["time_stamps"]) > 0:
                ts_rel = np.array(s["time_stamps"]) - session_t0
                descriptions = [item for sub in s["time_series"] for item in sub]
                ev_label = stream_name or stream_type_val or f"irregular_{len(events)}"
                events[ev_label] = (descriptions, ts_rel)
                logger.info(
                    "Stored irregular event stream '%s' (%d events)",
                    ev_label,
                    len(descriptions),
                )

        return raw, tier2, events, session_t0

    @staticmethod
    def _annotate_nan_regions(
        raw: mne.io.BaseRaw,
        nan_mask: np.ndarray,
        tgt_ts: np.ndarray,
        label: str,
        session_t0: float,
    ) -> None:
        """Add BAD annotations covering samples where an auxiliary stream was NaN."""
        padded = np.concatenate([[False], nan_mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s_idx, e_idx in zip(starts, ends):
            onset = tgt_ts[s_idx] - session_t0
            duration = tgt_ts[min(e_idx, len(tgt_ts) - 1)] - tgt_ts[s_idx]
            raw.annotations.append(onset, duration, f"BAD_{label}_missing")

    def _set_montage(self, raw: mne.io.BaseRaw) -> None:
        """Resolve and apply montage."""
        if isinstance(self.montage, mne.channels.DigMontage):
            logger.info("Using provided DigMontage object")
            raw.set_montage(self.montage, on_missing="warn")
        elif isinstance(self.montage, pathlib.Path):
            logger.info("Using custom montage from %s", self.montage)
            raw.set_montage(
                mne.channels.read_custom_montage(self.montage), on_missing="warn"
            )
        elif isinstance(self.montage, str):
            p = Path(self.montage)
            if p.exists():
                logger.info("Using custom montage from %s", self.montage)
                montage = mne.channels.read_custom_montage(str(p))
            else:
                logger.info("Using standard montage '%s'", self.montage)
                montage = mne.channels.make_standard_montage(self.montage)
            raw.set_montage(montage, on_missing="warn")

    def _apply_channel_selection(self, raw: mne.io.BaseRaw) -> None:
        """Drop/keep channels by name or channel type."""
        ch_type_constants = mne.io.get_channel_type_constants()

        def expand_types(names: list[str]) -> list[str]:
            expanded = []
            for n in names:
                if n in ch_type_constants:
                    expanded += [
                        ch["ch_name"]
                        for ch in raw.info["chs"]
                        if ch["kind"] == ch_type_constants[n]["kind"]
                    ]
                else:
                    expanded.append(n)
            return expanded

        to_drop: set[str] = set()

        if self.keep_channels:
            keep = set(expand_types(self.keep_channels))
            to_drop |= set(raw.ch_names) - keep

        if self.drop_channels:
            to_drop |= set(expand_types(self.drop_channels))

        if self.keep_channels and self.drop_channels:
            explicit_keep = {c for c in self.keep_channels if c in raw.ch_names}
            to_drop -= explicit_keep

        if to_drop:
            raw.drop_channels([ch for ch in to_drop if ch in raw.ch_names])
