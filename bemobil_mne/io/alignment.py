"""Timestamp-aware stream alignment for multimodal LSL/XDF recordings.

This module is intentionally free of MNE and other neuroscience-specific
dependencies so it can eventually be contributed upstream to pyxdf.

Public API
----------
align_stream_to_timestamps
    Align a (data, src_timestamps) pair to a set of target timestamps using
    timestamp-aware interpolation.  Handles NaN gaps and optional anti-aliasing.

Internal helpers (prefixed ``_``) are stable enough to import directly but are
not part of the public API.

Key references
--------------
[Butterworth 1930]    Butterworth, S. (1930). On the theory of filter amplifiers.
                      Wireless Engineer, 7, 536-541.
[Fritsch & Carlson 1980] Fritsch, F.N., & Carlson, R.E. (1980). Monotone piecewise
                      cubic interpolation. SIAM J Numer Anal, 17(2), 238-246.
                      https://doi.org/10.1137/0717021
[Shannon 1949]        Shannon, C.E. (1949). Communication in the presence of noise.
                      Proc. IRE, 37(1), 10-21.
                      https://doi.org/10.1109/JRPROC.1949.232969
[Widmann et al. 2015] Widmann, A., Schroger, E., & Maess, B. (2015). Digital filter
                      design for electrophysiological data - a practical approach.
                      J Neurosci Methods, 250, 34-46.
                      https://doi.org/10.1016/j.jneumeth.2014.08.002
[Gramfort et al. 2013] Gramfort, A., et al. (2013). MEG and EEG data analysis with
                      MNE-Python. Front Neurosci, 7, 267.
                      https://doi.org/10.3389/fnins.2013.00267
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.interpolate
import scipy.signal

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _compute_effective_srate(timestamps: np.ndarray) -> float:
    """Estimate effective sampling rate via median inter-sample interval."""
    if len(timestamps) < 2:
        return 0.0
    return float(1.0 / np.median(np.diff(timestamps)))


def _is_irregular(stream: dict) -> bool:
    """Return True if the stream has no fixed sampling rate (event-based).

    A stream is considered irregular when its ``nominal_srate`` is 0 *and*
    its computed effective rate is below 1 Hz.  This avoids false positives
    for streams that declare a nominal rate of 0 but still emit samples at a
    steady pace.
    """
    try:
        nominal = float(stream["info"]["nominal_srate"][0])
        effective = float(np.array(stream["info"].get("effective_srate", [0])).item())
    except (KeyError, ValueError, TypeError):
        nominal, effective = 0.0, 0.0
    return nominal == 0.0 and effective < 1.0


def _handle_nan_gaps(
    data: np.ndarray,
    timestamps: np.ndarray,
    method: str = "linear",
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Fill NaN gaps in *data* via interpolation; track long gaps.

    Stream-agnostic: any stream carrying NaN values (eye-tracking gaze during
    signal loss [Hershman et al. 2018], lost ECG packets, saturated EDA) is
    handled uniformly.  Short gaps are bridged rather than excluded because
    discarding every sample adjacent to a blink would reduce gaze data yield
    substantially.

    Parameters
    ----------
    data : (n_samples, n_channels)
        Raw data array, may contain NaN.
    timestamps : (n_samples,)
        Corresponding LSL timestamps.
    method : {'linear', 'pchip'}
        Interpolation method used to bridge NaN runs.
    max_gap_s : float or None
        NaN runs longer than this (seconds) are filled temporarily so
        interpolation can proceed, but their time intervals are returned so
        the caller can re-apply NaN in the aligned output.  ``None`` fills
        all gaps regardless of length.

    Returns
    -------
    filled_data : (n_samples, n_channels)
    long_gap_intervals : list of (t_start, t_end) in absolute LSL time
    """
    filled = data.copy().astype(float)
    long_gap_intervals: list[tuple[float, float]] = []

    for ch in range(data.shape[1]):
        col = filled[:, ch]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue

        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) == 0:
            continue  # entire channel NaN - nothing to interpolate

        # Identify contiguous NaN runs
        nan_indices = np.where(nan_mask)[0]
        breaks = np.where(np.diff(nan_indices) > 1)[0] + 1
        run_starts = np.concatenate([[nan_indices[0]], nan_indices[breaks]])
        run_ends = np.concatenate([nan_indices[breaks - 1], [nan_indices[-1]]])

        # Record long gaps once (channel 0) to avoid duplicates across channels
        if ch == 0 and max_gap_s is not None:
            for rs, re in zip(run_starts, run_ends):
                gap_duration = timestamps[re] - timestamps[rs]
                if gap_duration > max_gap_s:
                    long_gap_intervals.append(
                        (float(timestamps[rs]), float(timestamps[re]))
                    )

        # Interpolate across NaN runs
        if method == "pchip":
            interp = scipy.interpolate.PchipInterpolator(
                timestamps[valid_idx], col[valid_idx], extrapolate=False
            )
            filled_vals = interp(timestamps[nan_mask])
        else:  # linear
            interp = scipy.interpolate.interp1d(
                timestamps[valid_idx],
                col[valid_idx],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            filled_vals = interp(timestamps[nan_mask])

        # Only overwrite where interpolation produced a valid value
        col[nan_mask] = np.where(np.isnan(filled_vals), col[nan_mask], filled_vals)
        filled[:, ch] = col

    return filled, long_gap_intervals


def _apply_antialiasing(
    data: np.ndarray,
    src_sfreq: float,
    tgt_sfreq: float,
    order: int = 8,
) -> np.ndarray:
    """Low-pass filter *data* at 90 % of the target Nyquist before downsampling.

    Only applied when ``tgt_sfreq < src_sfreq``.  The 10 % guard margin below
    Nyquist provides a transition band that prevents ringing at the exact
    cutoff.  An 8th-order zero-phase Butterworth (``filtfilt``) gives
    ~48 dB/octave rolloff with a maximally flat passband and no phase
    distortion [Butterworth 1930; Widmann et al. 2015].  Zero-phase
    (forward + backward) filtering preserves event-related latencies, which
    is critical for multimodal alignment [Widmann et al. 2015].
    """
    if tgt_sfreq >= src_sfreq:
        return data
    cutoff_ratio = min((tgt_sfreq / 2.0 * 0.9) / (src_sfreq / 2.0), 0.99)
    b, a = scipy.signal.butter(order, cutoff_ratio, btype="low")
    return scipy.signal.filtfilt(b, a, data, axis=0)


# ---------------------------------------------------------------------------
# Stim channel helper
# ---------------------------------------------------------------------------


def _align_stim_channels(
    data: np.ndarray,
    src_ts: np.ndarray,
    tgt_ts: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Align discrete-valued (stim/trigger) channels without interpolation.

    For each output sample window, selects the first non-zero source value
    within that window, falling back to the first value if all are zero.
    This mirrors the approach used in ``mne.filter._resample_stim_channels``
    [Gramfort et al. 2013] but is timestamp-aware rather than ratio-based,
    so it works correctly when source and target clocks are not integer
    multiples of each other.

    The algorithm is fully vectorised via ``np.searchsorted`` and
    ``np.lexsort`` and runs in O(n_src * log(n_src)) time.

    Parameters
    ----------
    data : (n_samples, n_channels)
    src_ts : (n_samples,)
    tgt_ts : (n_target,)
    fill_value : float
        Value for output windows that contain no source samples (default 0).

    Returns
    -------
    aligned : (n_target, n_channels)
    """
    n_tgt = len(tgt_ts)
    n_channels = data.shape[1]
    aligned = np.full((n_tgt, n_channels), fill_value, dtype=float)

    # Assign each source sample to the target bin it falls in (left edge).
    bin_idx = np.searchsorted(tgt_ts, src_ts, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_tgt)
    if not valid.any():
        return aligned

    data_v = data[valid].astype(float)
    bins_v = bin_idx[valid]
    src_pos = np.where(valid)[0]  # original position for tie-breaking

    # Sort key: (bin, not_nonzero, src_position)
    # - Groups by bin
    # - Within a bin: non-zero samples (key=0) before zero samples (key=1)
    # - Among equal non-zero/zero: earlier source position wins
    is_nonzero = (data_v != 0).any(axis=1).astype(np.intp)
    order = np.lexsort((src_pos, 1 - is_nonzero, bins_v))

    data_sorted = data_v[order]
    bins_sorted = bins_v[order]

    # First occurrence of each bin after sorting = winner for that bin
    _, first = np.unique(bins_sorted, return_index=True)
    aligned[bins_sorted[first]] = data_sorted[first]

    return aligned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_stream_to_timestamps(
    data: np.ndarray,
    src_ts: np.ndarray,
    tgt_ts: np.ndarray,
    method: Literal["linear", "pchip", "sinc", "nearest", "stim"] = "pchip",
    fill_value: float = np.nan,
    nan_gap_method: Literal["linear", "pchip"] = "linear",
    max_nan_gap_s: float | None = None,
) -> np.ndarray:
    """Align a data stream to a set of target timestamps.

    The core alignment primitive for multimodal XDF recordings.  Rather than
    resampling by target *length* (which discards timing information), this
    function uses the actual LSL timestamps of both the source stream and the
    target grid to evaluate the signal at the correct instants.

    pyxdf corrects clock offsets between devices before returning timestamps,
    so the ``src_ts`` values passed here are already on a shared reference
    clock.

    Parameters
    ----------
    data : (n_samples, n_channels) or (n_samples,)
        Source data array.
    src_ts : (n_samples,)
        LSL timestamps of the source samples (clock-corrected by pyxdf).
    tgt_ts : (n_target,)
        Target LSL timestamps at which to evaluate the stream.  Typically a
        uniform grid derived from the primary stream's time range.
    method : {'linear', 'pchip', 'sinc', 'nearest', 'stim'}
        Alignment strategy.

        ``'linear'``
            Fast piecewise-linear interpolation.  Sufficient when the source
            and target rates are similar and the signal is smooth.

        ``'pchip'``
            Piecewise Cubic Hermite Interpolating Polynomial [Fritsch &
            Carlson 1980].  Preserves local monotonicity and avoids the
            overshoot of natural cubic splines.  Preferred for slow
            physiological signals (EDA, pupil diameter) and for large
            up-sampling ratios.

        ``'sinc'``
            Applies an 8th-order zero-phase Butterworth anti-aliasing filter
            [Butterworth 1930; Widmann et al. 2015] at 90 % of the target
            Nyquist frequency *before* PCHIP interpolation.  Use whenever
            the source rate exceeds the target rate (e.g. ECG or EMG at
            1000-2000 Hz aligned to a 500 Hz common grid) to prevent aliasing
            of high-frequency energy into the passband [Shannon 1949].

        ``'nearest'``
            Zero-order hold: each output sample takes the value of the
            nearest source sample in time.  No interpolation is performed.
            Appropriate for discrete-valued channels where intermediate
            values are meaningless (e.g. a button-box channel that only
            ever holds 0 or 1).

        ``'stim'``
            Timestamp-aware trigger/stimulus resampling [Gramfort et al.
            2013].  For each output window, selects the first *non-zero*
            source value within that window, falling back to the first value
            if all are zero.  Preserves trigger pulses that might otherwise
            be averaged away or missed by nearest-neighbor selection.  Use
            for trigger channels, button boxes, or any channel encoding
            sparse events as non-zero pulses in an otherwise-zero baseline.

    fill_value : float
        Value assigned to target timestamps outside the source time range.
        Defaults to ``np.nan`` so out-of-range regions are clearly marked
        rather than silently extrapolated.  For ``'stim'`` the effective
        fill defaults to 0 (no event).
    nan_gap_method : {'linear', 'pchip'}
        Interpolation method used to bridge NaN runs within the source data
        before alignment.  Not applied for ``'nearest'`` or ``'stim'``.
    max_nan_gap_s : float or None
        NaN runs in the source data longer than this (seconds) are preserved
        as NaN in the output.  Shorter gaps are filled by ``nan_gap_method``.
        ``None`` fills all gaps.  Not applied for ``'nearest'`` or ``'stim'``.

    Returns
    -------
    aligned : (n_target, n_channels)
        Data evaluated at *tgt_ts*.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # nearest and stim bypass interpolation entirely - no NaN handling needed
    if method == "nearest":
        idx = np.searchsorted(src_ts, tgt_ts, side="left")
        idx = np.clip(idx, 0, len(src_ts) - 1)
        left = np.clip(idx - 1, 0, len(src_ts) - 1)
        use_left = np.abs(src_ts[left] - tgt_ts) < np.abs(src_ts[idx] - tgt_ts)
        idx = np.where(use_left, left, idx)
        out = data[idx].copy()
        out_of_range = (tgt_ts < src_ts[0]) | (tgt_ts > src_ts[-1])
        out[out_of_range] = fill_value
        return out

    if method == "stim":
        stim_fill = 0.0 if np.isnan(fill_value) else fill_value
        return _align_stim_channels(data, src_ts, tgt_ts, fill_value=stim_fill)

    # 1. Fill NaN gaps in source; track long ones for re-application
    long_gap_intervals: list[tuple[float, float]] = []
    if np.isnan(data).any():
        data, long_gap_intervals = _handle_nan_gaps(
            data, src_ts, method=nan_gap_method, max_gap_s=max_nan_gap_s
        )

    # 2. Anti-aliasing filter (sinc only, downsampling only)
    if method == "sinc":
        src_sfreq = _compute_effective_srate(src_ts)
        tgt_sfreq = _compute_effective_srate(tgt_ts)
        if src_sfreq > 0 and tgt_sfreq > 0:
            data = _apply_antialiasing(data, src_sfreq, tgt_sfreq)

    # 3. Interpolate at target timestamps
    n_channels = data.shape[1]
    aligned = np.full((len(tgt_ts), n_channels), fill_value, dtype=float)
    use_pchip = method in ("pchip", "sinc")

    for ch in range(n_channels):
        col = data[:, ch]
        valid = ~np.isnan(col)
        if not valid.any():
            continue

        x, y = src_ts[valid], col[valid]

        if use_pchip:
            interp = scipy.interpolate.PchipInterpolator(x, y, extrapolate=False)
            vals = interp(tgt_ts)
            outside = (tgt_ts < x[0]) | (tgt_ts > x[-1])
            vals = np.where(np.isnan(vals) | outside, fill_value, vals)
        else:
            interp = scipy.interpolate.interp1d(
                x, y, kind="linear", bounds_error=False, fill_value=fill_value
            )
            vals = interp(tgt_ts)

        aligned[:, ch] = vals

    # 4. Re-apply NaN for long gaps that should stay missing
    for t_start, t_end in long_gap_intervals:
        gap_mask = (tgt_ts >= t_start) & (tgt_ts <= t_end)
        aligned[gap_mask, :] = np.nan

    return aligned
