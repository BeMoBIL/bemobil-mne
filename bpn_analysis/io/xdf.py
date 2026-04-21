"""Simplified XDF loader for EEG data."""

# %% Imports

from __future__ import annotations

import logging
import pathlib
from pathlib import Path

import mne
import numpy as np
import pyxdf
import scipy.signal

# %% Constants & Settings

logger = logging.getLogger(__name__)

_MICROVOLT_UNITS = frozenset(("microvolt", "microvolts", "µv", "μv", "uv"))
_VOLT_UNITS = frozenset(("v", "volt", "volts"))

# %% Helper functions


def _unit_scale(unit: str, ch_name: str = "") -> float:
    """Return scale factor to convert *unit* to SI volts (case-insensitive).

    Emits a warning for unrecognised units so silent identity scaling is visible.
    """
    normed = unit.strip().lower()
    if normed in _MICROVOLT_UNITS:
        return 1e-6
    if normed not in _VOLT_UNITS and normed not in ("na", ""):
        logger.warning(
            "Unrecognised unit '%s' for channel '%s' — assuming volts (no scaling applied).",
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
            f"Multiple streams matched criteria (name={name}, "
            f"type={stream_type}). Using first."
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

    return labels, types, units


def _crop_stream(
    stream: dict, start_time: float, end_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crop stream to [start_time, end_time]."""
    ts = np.array(stream["time_stamps"])
    data = np.array(stream["time_series"])
    mask = (ts >= start_time) & (ts <= end_time)
    return data[mask], ts[mask]


def _resample_stream(
    data: np.ndarray, timestamps: np.ndarray, target_len: int, target_ts: np.ndarray
) -> np.ndarray:
    """Resample stream data to target_len samples using scipy."""
    return scipy.signal.resample(data, target_len, axis=0)


# %% Main class


class XDFLoader:
    """
    Simplified EEG loader for XDF files.

    Parameters
    ----------
    eeg_stream_name : str, optional
        Name of the EEG stream to load.
    eeg_stream_type : str
        Type of the EEG stream. Default: "EEG".
    eeg_source_id : str, optional
        Source ID of the EEG stream.
    marker_stream_types : list[str]
        Types to search for marker streams. Default: ["Markers", "Logging", "Notes"].
    special_streams : dict, optional
        Dict of {label: {"name": ..., "type": ...}} for aux streams (e.g. eye-tracking).
        These will be resampled to EEG rate and appended as channels.
    montage : str, pathlib.Path, or DigMontage, optional
        Montage to set for EEG channels. Can be a standard montage name, a path to a
        custom montage file, or a DigMontage object.
    old_reference : str, optional
        Reference channel to add back before setting montage (e.g. "FCz").
    keep_channels : list[str], optional
        Channel names or types to keep (all others dropped).
    drop_channels : list[str], optional
        Channel names or types to drop.
    target_sfreq : float, optional
        Resample EEG to this frequency after loading.
    """

    def __init__(
        self,
        eeg_stream_name: str | None = None,
        eeg_stream_type: str = "EEG",
        eeg_source_id: str | None = None,
        marker_stream_types: list[str] | None = None,
        special_streams: dict[str, dict] | None = None,
        montage: str | mne.channels.DigMontage | None = None,
        old_reference: str | None = None,
        keep_channels: list[str] | None = None,
        drop_channels: list[str] | None = None,
        target_sfreq: float | None = None,
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
        self.montage = montage
        self.old_reference = old_reference
        self.keep_channels = keep_channels
        self.drop_channels = drop_channels
        self.target_sfreq = target_sfreq

    def load(self, path: str | Path) -> mne.io.BaseRaw:
        """
        Load an XDF file and return an MNE Raw object.

        Parameters
        ----------
        path : str or Path
            Path to the XDF file.

        Returns
        -------
        mne.io.RawArray
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"XDF file not found: {path}")
        if path.suffix.lower() != ".xdf":
            raise ValueError(f"Expected an XDF file, got: {path.suffix}")

        streams, _ = pyxdf.load_xdf(str(path))
        logger.info(f"Loaded {len(streams)} streams from {path.name}")
        for s in streams:
            info = s["info"]
            logger.info(
                f"  Stream: name='{info.get('name', [''])[0]}' "
                f"type='{info.get('type', [''])[0]}' "
                f"source_id='{info.get('source_id', [''])[0]}'"
            )

        raw = self._build_raw(streams)

        if self.old_reference:
            raw.add_reference_channels(self.old_reference)

        if self.montage:
            self._set_montage(raw)

        if self.target_sfreq:
            raw.resample(self.target_sfreq)

        self._apply_channel_selection(raw)

        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_raw(self, streams: list[dict]) -> mne.io.RawArray:
        """Build MNE RawArray from EEG stream, attach markers and special streams."""
        # --- EEG stream ---
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
        sfreq = float(np.array(eeg_stream["info"]["effective_srate"]).item())
        scale = np.array([_unit_scale(u, ch) for u, ch in zip(units, labels)])

        start_time = eeg_stream["time_stamps"][0]
        end_time = eeg_stream["time_stamps"][-1]

        # Narrow time range to overlap with special streams
        special_stream_objs = {}
        for label, spec in self.special_streams.items():
            s = _match_stream(
                streams, name=spec.get("name"), stream_type=spec.get("type")
            )
            if s is None:
                logger.warning(f"Special stream '{label}' not found, skipping.")
                continue
            if len(s["time_stamps"]) >= 2:
                start_time = max(start_time, s["time_stamps"][0])
                end_time = min(end_time, s["time_stamps"][-1])
            special_stream_objs[label] = s

        eeg_data, _ = _crop_stream(eeg_stream, start_time, end_time)
        info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=types)
        raw = mne.io.RawArray((eeg_data * scale).T, info, verbose=False)

        # --- Special streams ---
        target_len = len(raw.times)
        target_ts = np.linspace(start_time, end_time, target_len)

        for label, s in special_stream_objs.items():
            data, ts = _crop_stream(s, start_time, end_time)
            resampled = _resample_stream(data, ts, target_len, target_ts)
            ch_labels, ch_types, _ = _extract_channel_info(s)
            aux_info = mne.create_info(
                ch_names=ch_labels, sfreq=sfreq, ch_types=ch_types
            )
            aux_raw = mne.io.RawArray(resampled.T, aux_info, verbose=False)
            try:
                raw.add_channels([aux_raw], force_update_info=True)
            except ValueError:
                aux_raw.rename_channels(
                    {ch: f"{label}_{ch}" for ch in aux_raw.ch_names}
                )
                raw.add_channels([aux_raw], force_update_info=True)
            logger.info(f"Added special stream '{label}' ({len(ch_labels)} channels)")

        # --- Marker streams ---
        for stream_type in self.marker_stream_types:
            matched = _match_stream(
                streams, stream_type=stream_type, allow_multiple=True
            )
            for s in matched:
                onsets = np.array(s["time_stamps"]) - start_time
                descriptions = [item for sub in s["time_series"] for item in sub]
                raw.annotations.append(onsets, [0.0] * len(onsets), descriptions)
                logger.info(
                    f"Added {len(descriptions)} markers from stream "
                    f"'{s['info'].get('name', ['?'])[0]}'"
                )

        return raw

    def _set_montage(self, raw: mne.io.BaseRaw) -> None:
        """Resolve and apply montage."""
        if isinstance(self.montage, mne.channels.DigMontage):
            logger.info("Using provided DigMontage object")
            raw.set_montage(self.montage, on_missing="warn")
        elif isinstance(self.montage, pathlib.Path):
            logger.info(f"Using custom montage from {self.montage}")
            raw.set_montage(
                mne.channels.read_custom_montage(self.montage), on_missing="warn"
            )
        elif isinstance(self.montage, str):
            p = Path(self.montage)
            if p.exists():
                logger.info(f"Using custom montage from {self.montage}")
                montage = mne.channels.read_custom_montage(str(p))
            else:
                logger.info(f"Using standard montage '{self.montage}'")
                montage = mne.channels.make_standard_montage(self.montage)
            raw.set_montage(montage, on_missing="warn")

    def _apply_channel_selection(self, raw: mne.io.BaseRaw) -> None:
        """Drop/keep channels by name or channel type."""
        ch_type_constants = mne.io.get_channel_type_constants()

        def expand_types(names: list[str]) -> list[str]:
            """Expand any channel type strings to actual channel names."""
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

        to_drop = set()

        if self.keep_channels:
            keep = set(expand_types(self.keep_channels))
            to_drop |= set(raw.ch_names) - keep

        if self.drop_channels:
            to_drop |= set(expand_types(self.drop_channels))

        # keep_channels takes priority for explicitly named channels
        if self.keep_channels and self.drop_channels:
            explicit_keep = set(c for c in self.keep_channels if c in raw.ch_names)
            to_drop -= explicit_keep

        if to_drop:
            raw.drop_channels([ch for ch in to_drop if ch in raw.ch_names])
