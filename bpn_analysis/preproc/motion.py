"""Rigid body motion capture processing utilities.

Implements kinematics processing equivalent to BeMoBIL's
``bemobil_process_all_motion.m`` pipeline:

1. Identify rigid body channel groups from naming conventions (quaternion or
   Euler suffixes).
2. Optionally low-pass filter the position / orientation signals.
3. Convert unit quaternions to Euler angles (roll / pitch / yaw) when needed.
4. Compute first derivative (velocity) with optional low-pass filtering.
5. Compute second derivative (acceleration) with optional low-pass filtering.

All operations work on :class:`mne.io.BaseRaw` objects.  Rigid body channels
are expected to follow one of two naming conventions:

* **Quaternion**: ``<rigid_body>_quat_x``, ``<rigid_body>_quat_y``,
  ``<rigid_body>_quat_z``, ``<rigid_body>_quat_w``
* **Euler angles**: ``<rigid_body>_eul_x`` (roll), ``<rigid_body>_eul_y``
  (pitch), ``<rigid_body>_eul_z`` (yaw)

Position channels follow the convention ``<rigid_body>_pos_x``,
``<rigid_body>_pos_y``, ``<rigid_body>_pos_z``.
"""

# %% Imports

from __future__ import annotations

import logging
import re

import mne
import numpy as np

# %% Settings & Constants

LOGGER = logging.getLogger(__name__)

# Suffix patterns used to identify rigid body sub-channels
_QUAT_SUFFIXES = ("_quat_x", "_quat_y", "_quat_z", "_quat_w")
_EUL_SUFFIXES = ("_eul_x", "_eul_y", "_eul_z")
_POS_SUFFIXES = ("_pos_x", "_pos_y", "_pos_z")

_QUAT_RE = re.compile(r"^(.+)_quat_[xyzw]$")
_EUL_RE = re.compile(r"^(.+)_eul_[xyz]$")
_POS_RE = re.compile(r"^(.+)_pos_[xyz]$")


# %% Functions


def find_rigid_bodies(raw):
    """Detect rigid body names from channel naming conventions.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording whose channel names to inspect.

    Returns
    -------
    rb_names : list of str
        Unique rigid body identifiers found in the channel list.
    """
    names: set[str] = set()
    for ch in raw.ch_names:
        for pat in (_QUAT_RE, _EUL_RE, _POS_RE):
            m = pat.match(ch)
            if m:
                names.add(m.group(1))
                break
    return sorted(names)


def process_rigid_body(
    raw,
    rb_names=None,
    lowpass_orient=8.0,
    lowpass_pos=8.0,
    lowpass_deriv=24.0,
    compute_velocity=True,
    compute_acceleration=True,
    sfreq_target=None,
):
    """Process rigid body motion capture streams.

    For each rigid body found in *raw*:

    1. Optionally low-pass filter the orientation (and position) signals.
    2. Convert quaternions to Euler angles (ZYX convention = yaw-pitch-roll)
       if quaternion channels are present.  The source quaternion channels are
       dropped and replaced by Euler angle channels.
    3. Compute the first derivative (velocity) via finite differences, with an
       optional low-pass filter applied after differentiation.
    4. Compute the second derivative (acceleration) in the same way.

    Derivative channels are appended to the returned raw object with the
    naming scheme ``<rb>_vel_<axis>`` and ``<rb>_acc_<axis>``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Source recording.  Motion channels should be of type ``"misc"``
        (which is the default when XDFLoader imports non-EEG streams).
    rb_names : list of str | None
        Rigid body identifiers to process.  If ``None``, all identifiers
        found by :func:`find_rigid_bodies` are processed.
    lowpass_orient : float | None
        Low-pass cutoff (Hz) applied to orientation channels.  ``None``
        skips this filter.
    lowpass_pos : float | None
        Low-pass cutoff (Hz) applied to position channels.  ``None`` skips.
    lowpass_deriv : float | None
        Low-pass cutoff (Hz) applied to velocity and acceleration channels
        *after* differentiation.  ``None`` skips.
    compute_velocity : bool
        Whether to compute and append velocity channels.
    compute_acceleration : bool
        Whether to compute and append acceleration channels.
    sfreq_target : float | None
        If not ``None``, resample motion channels to this rate before
        differentiation.  Useful to match the EEG sample rate.

    Returns
    -------
    raw_out : mne.io.Raw
        Copy of *raw* with quaternion channels replaced by Euler channels and
        velocity / acceleration channels appended.
    rb_info : dict
        Per-rigid-body processing summary:
        ``{rb_name: {"mode": "quat"|"euler", "channels": list[str]}}``.
    """
    raw_out = raw.copy()
    sfreq = raw_out.info["sfreq"]

    if rb_names is None:
        rb_names = find_rigid_bodies(raw_out)

    if not rb_names:
        LOGGER.warning("process_rigid_body: no rigid body channels found.")
        return raw_out, {}

    rb_info: dict = {}

    for rb in rb_names:
        LOGGER.info(f"Processing rigid body: {rb}")
        info_entry: dict = {}

        # ----------------------------------------------------------------
        # Detect channel mode
        # ----------------------------------------------------------------
        quat_chs = [
            f"{rb}{s}" for s in _QUAT_SUFFIXES if f"{rb}{s}" in raw_out.ch_names
        ]
        eul_chs = [f"{rb}{s}" for s in _EUL_SUFFIXES if f"{rb}{s}" in raw_out.ch_names]
        pos_chs = [f"{rb}{s}" for s in _POS_SUFFIXES if f"{rb}{s}" in raw_out.ch_names]

        if not quat_chs and not eul_chs:
            LOGGER.warning(f"  {rb}: no orientation channels found, skipping.")
            continue

        orient_chs: list[str]
        mode: str

        if quat_chs:
            mode = "quat"
            # ----------------------------------------------------------------
            # Quaternion → Euler conversion
            # ----------------------------------------------------------------
            q_idx = [raw_out.ch_names.index(ch) for ch in quat_chs]
            # Order: x, y, z, w
            q_data = raw_out.get_data(picks=q_idx)  # (4, n_times)
            # Normalise quaternions row-wise
            q_norm = np.linalg.norm(q_data, axis=0, keepdims=True)
            q_data = q_data / np.where(q_norm > 1e-10, q_norm, 1.0)

            euler_data = _quat_to_euler_zyx(q_data)  # (3, n_times)

            # Build new Euler channel info
            new_ch_names = [f"{rb}_eul_x", f"{rb}_eul_y", f"{rb}_eul_z"]
            new_info = mne.create_info(
                ch_names=new_ch_names,
                sfreq=sfreq,
                ch_types=["misc"] * 3,
            )
            euler_raw = mne.io.RawArray(euler_data, new_info, verbose=False)
            euler_raw.set_meas_date(raw_out.info["meas_date"])

            # Drop original quaternion channels and append Euler channels
            raw_out.drop_channels(quat_chs)
            raw_out.add_channels([euler_raw], force_update_info=True)

            orient_chs = new_ch_names
            info_entry["mode"] = "quat->euler"

        else:
            mode = "euler"
            orient_chs = eul_chs
            info_entry["mode"] = "euler"

        # ----------------------------------------------------------------
        # Low-pass filter orientation
        # ----------------------------------------------------------------
        if lowpass_orient is not None:
            orient_idx = [raw_out.ch_names.index(ch) for ch in orient_chs]
            raw_out._data[orient_idx] = _lowpass_array(
                raw_out._data[orient_idx], sfreq, lowpass_orient
            )

        # ----------------------------------------------------------------
        # Low-pass filter position
        # ----------------------------------------------------------------
        if pos_chs and lowpass_pos is not None:
            pos_idx = [raw_out.ch_names.index(ch) for ch in pos_chs]
            raw_out._data[pos_idx] = _lowpass_array(
                raw_out._data[pos_idx], sfreq, lowpass_pos
            )

        # ----------------------------------------------------------------
        # Derivatives
        # ----------------------------------------------------------------
        all_src_chs = orient_chs + pos_chs
        all_ch_idx = [raw_out.ch_names.index(ch) for ch in all_src_chs]
        src_data = raw_out.get_data(picks=all_ch_idx)  # (n_chs, n_times)

        appended_chs: list[str] = list(orient_chs) + list(pos_chs)

        if compute_velocity:
            vel_data = np.gradient(src_data, 1.0 / sfreq, axis=1)
            if lowpass_deriv is not None:
                vel_data = _lowpass_array(vel_data, sfreq, lowpass_deriv)
            vel_names = [_src_to_deriv(ch, "_vel") for ch in all_src_chs]
            vel_raw = mne.io.RawArray(
                vel_data,
                mne.create_info(vel_names, sfreq, ["misc"] * len(vel_names)),
                verbose=False,
            )
            vel_raw.set_meas_date(raw_out.info["meas_date"])
            raw_out.add_channels([vel_raw], force_update_info=True)
            appended_chs.extend(vel_names)

            if compute_acceleration:
                acc_data = np.gradient(vel_data, 1.0 / sfreq, axis=1)
                if lowpass_deriv is not None:
                    acc_data = _lowpass_array(acc_data, sfreq, lowpass_deriv)
                acc_names = [_src_to_deriv(ch, "_acc") for ch in all_src_chs]
                acc_raw = mne.io.RawArray(
                    acc_data,
                    mne.create_info(acc_names, sfreq, ["misc"] * len(acc_names)),
                    verbose=False,
                )
                acc_raw.set_meas_date(raw_out.info["meas_date"])
                raw_out.add_channels([acc_raw], force_update_info=True)
                appended_chs.extend(acc_names)

        info_entry["channels"] = appended_chs
        rb_info[rb] = info_entry
        LOGGER.info(f"  {rb}: processed ({mode}), channels: {appended_chs}")

    return raw_out, rb_info


def split_by_rigid_body(raw, rb_names=None):
    """Return a dict of per-rigid-body Raw sub-objects.

    Picks all channels whose name starts with each rigid body identifier
    (including velocity / acceleration channels appended by
    :func:`process_rigid_body`).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording (typically the output of :func:`process_rigid_body`).
    rb_names : list of str | None
        Rigid body names.  Detected automatically when ``None``.

    Returns
    -------
    dict of {str: mne.io.Raw}
    """
    if rb_names is None:
        rb_names = find_rigid_bodies(raw)
    out = {}
    for rb in rb_names:
        chs = [ch for ch in raw.ch_names if ch.startswith(rb + "_")]
        if chs:
            out[rb] = raw.copy().pick(chs)
    return out


# %% Private helpers


def _quat_to_euler_zyx(q):
    """Convert unit quaternions to ZYX Euler angles (yaw-pitch-roll).

    Parameters
    ----------
    q : ndarray, shape (4, n_times)
        Rows are qx, qy, qz, qw.

    Returns
    -------
    euler : ndarray, shape (3, n_times)
        Rows are roll (x), pitch (y), yaw (z) in radians.
    """
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    # Roll (x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y) - clamped to avoid arcsin domain errors from floating point
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.vstack([roll, pitch, yaw])


def _lowpass_array(data, sfreq, cutoff):
    """Apply a zero-phase FIR low-pass filter to each row of *data*."""
    from mne.filter import filter_data

    return filter_data(
        data.astype(np.float64),
        sfreq=sfreq,
        l_freq=None,
        h_freq=float(cutoff),
        method="fir",
        fir_window="hamming",
        verbose=False,
    )


def _src_to_deriv(ch_name, suffix):
    """Rename a source channel to its derivative counterpart.

    ``head_eul_x`` → ``head_vel_x`` (for suffix ``"_vel"``).
    """
    # Replace the last segment type indicator (eul / pos / vel) with suffix
    for segment in ("_eul_", "_pos_", "_vel_", "_quat_"):
        if segment in ch_name:
            axis = ch_name.split(segment)[-1]
            rb = ch_name.split(segment)[0]
            return f"{rb}{suffix}_{axis}"
    return f"{ch_name}{suffix}"
