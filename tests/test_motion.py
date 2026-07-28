"""Tests for bemobil_mne.preproc.motion."""

# %% Imports

import mne
import numpy as np
import pytest

from bemobil_mne.preproc.motion import (
    _lowpass_array,
    _quat_to_euler_zyx,
    _src_to_deriv,
    find_rigid_bodies,
    process_rigid_body,
    split_by_rigid_body,
)

# %% find_rigid_bodies


def test_find_rigid_bodies_detects_quat_channels(motion_raw):
    """Detect rigid bodies from quaternion channels."""
    bodies = find_rigid_bodies(motion_raw)
    assert "head" in bodies


def test_find_rigid_bodies_detects_euler_channels(motion_raw):
    """Detect rigid bodies from Euler angle channels."""
    bodies = find_rigid_bodies(motion_raw)
    assert "hand" in bodies


def test_find_rigid_bodies_detects_pos_channels(motion_raw):
    """Detect all rigid bodies with position channels."""
    # Both head and hand have pos channels
    bodies = find_rigid_bodies(motion_raw)
    assert set(bodies) == {"head", "hand"}


def test_find_rigid_bodies_empty_when_no_motion():
    """Return empty list when no motion channels present."""
    info = mne.create_info(["Cz", "Fz"], sfreq=250.0, ch_types="eeg")
    data = np.zeros((2, 250))
    raw = mne.io.RawArray(data, info, verbose=False)
    assert find_rigid_bodies(raw) == []


def test_find_rigid_bodies_sorted(motion_raw):
    """Return body names in sorted order."""
    bodies = find_rigid_bodies(motion_raw)
    assert bodies == sorted(bodies)


# %% process_rigid_body


def test_process_rigid_body_replaces_quat_with_euler(motion_raw):
    """Replace quaternion channels with Euler angle channels."""
    raw_out, rb_info = process_rigid_body(motion_raw)
    # Quaternion channels should be gone
    for quat_ch in ("head_quat_x", "head_quat_y", "head_quat_z", "head_quat_w"):
        assert quat_ch not in raw_out.ch_names
    # Euler channels should be present
    for eul_ch in ("head_eul_x", "head_eul_y", "head_eul_z"):
        assert eul_ch in raw_out.ch_names


def test_process_rigid_body_adds_velocity_channels(motion_raw):
    """Add velocity channels when compute_velocity is True."""
    raw_out, _ = process_rigid_body(motion_raw, compute_velocity=True)
    # naming: <rb>_<type>_vel_<axis>  (type preserved to avoid eul/pos collisions)
    assert "head_eul_vel_x" in raw_out.ch_names
    assert "head_pos_vel_x" in raw_out.ch_names
    assert "hand_eul_vel_x" in raw_out.ch_names


def test_process_rigid_body_adds_acceleration_channels(motion_raw):
    """Add acceleration channels when compute_acceleration is True."""
    raw_out, _ = process_rigid_body(
        motion_raw, compute_velocity=True, compute_acceleration=True
    )
    assert "head_eul_acc_x" in raw_out.ch_names
    assert "head_pos_acc_x" in raw_out.ch_names
    assert "hand_eul_acc_x" in raw_out.ch_names


def test_process_rigid_body_no_velocity(motion_raw):
    """Omit velocity and acceleration when compute_velocity is False."""
    raw_out, _ = process_rigid_body(motion_raw, compute_velocity=False)
    assert all("_vel_" not in ch for ch in raw_out.ch_names)
    assert all("_acc_" not in ch for ch in raw_out.ch_names)


def test_process_rigid_body_no_acceleration(motion_raw):
    """Omit acceleration channels when compute_acceleration is False."""
    raw_out, _ = process_rigid_body(
        motion_raw, compute_velocity=True, compute_acceleration=False
    )
    assert any("_vel_" in ch for ch in raw_out.ch_names)
    assert all("_acc_" not in ch for ch in raw_out.ch_names)


def test_process_rigid_body_rb_info_keys(motion_raw):
    """Verify rb_info contains expected keys per body."""
    _, rb_info = process_rigid_body(motion_raw)
    assert set(rb_info.keys()) == {"head", "hand"}
    assert "mode" in rb_info["head"]
    assert "channels" in rb_info["head"]


def test_process_rigid_body_head_mode_quat_to_euler(motion_raw):
    """Record quat->euler mode for head rigid body."""
    _, rb_info = process_rigid_body(motion_raw)
    assert rb_info["head"]["mode"] == "quat->euler"


def test_process_rigid_body_hand_mode_euler(motion_raw):
    """Record euler mode for hand rigid body."""
    _, rb_info = process_rigid_body(motion_raw)
    assert rb_info["hand"]["mode"] == "euler"


def test_process_rigid_body_subset(motion_raw):
    """Processing only one body leaves the other untouched."""
    raw_out, rb_info = process_rigid_body(motion_raw, rb_names=["head"])
    assert set(rb_info.keys()) == {"head"}


def test_process_rigid_body_warns_on_empty(caplog):
    """Log warning when no rigid body channels found."""
    info = mne.create_info(["Cz"], sfreq=250.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 250)), info, verbose=False)
    import logging

    with caplog.at_level(logging.WARNING):
        process_rigid_body(raw)
    assert "no rigid body channels" in caplog.text.lower()


def test_process_rigid_body_preserves_n_times(motion_raw):
    """Preserve the number of time points."""
    n_times_before = motion_raw.get_data().shape[1]
    raw_out, _ = process_rigid_body(motion_raw)
    assert raw_out.get_data().shape[1] == n_times_before


# %% split_by_rigid_body


def test_split_by_rigid_body_returns_dict(motion_raw):
    """Return a dict keyed by rigid body."""
    raw_proc, _ = process_rigid_body(motion_raw)
    split = split_by_rigid_body(raw_proc)
    assert isinstance(split, dict)
    assert "head" in split
    assert "hand" in split


def test_split_by_rigid_body_channels_are_subset(motion_raw):
    """Verify each sub-Raw has only its body channels."""
    raw_proc, _ = process_rigid_body(motion_raw)
    split = split_by_rigid_body(raw_proc)
    for rb, sub in split.items():
        for ch in sub.ch_names:
            assert ch.startswith(rb + "_")


def test_split_by_rigid_body_explicit_names(motion_raw):
    """Restrict output to explicitly requested body names."""
    raw_proc, _ = process_rigid_body(motion_raw)
    split = split_by_rigid_body(raw_proc, rb_names=["head"])
    assert set(split.keys()) == {"head"}


# %% _quat_to_euler_zyx


def test_quat_to_euler_identity():
    """Map identity quaternion to zero Euler angles."""
    n = 10
    q = np.zeros((4, n))
    q[3] = 1.0  # qw = 1
    euler = _quat_to_euler_zyx(q)
    np.testing.assert_allclose(euler, 0.0, atol=1e-10)


def test_quat_to_euler_90_deg_yaw():
    """Convert 90-degree yaw quaternion to Euler angles."""
    angle = np.pi / 2
    q = np.array([[0.0], [0.0], [np.sin(angle / 2)], [np.cos(angle / 2)]])
    euler = _quat_to_euler_zyx(q)
    # yaw = row 2
    np.testing.assert_allclose(euler[2, 0], angle, atol=1e-6)
    np.testing.assert_allclose(euler[:2, 0], 0.0, atol=1e-6)


def test_quat_to_euler_output_shape():
    """Return array of shape (3, n)."""
    n = 50
    q = np.zeros((4, n))
    q[3] = 1.0
    euler = _quat_to_euler_zyx(q)
    assert euler.shape == (3, n)


# %% _lowpass_array


def test_lowpass_array_preserves_shape():
    """Return array with same shape as input."""
    sfreq = 250.0
    data = np.random.default_rng(0).standard_normal((4, int(sfreq * 5)))
    out = _lowpass_array(data, sfreq, cutoff=8.0)
    assert out.shape == data.shape


def test_lowpass_array_attenuates_high_freq():
    """Attenuate high-frequency content above cutoff."""
    sfreq = 250.0
    n = int(sfreq * 10)
    t = np.arange(n) / sfreq
    # signal = low (2 Hz) + high (80 Hz) - only 1 channel
    signal = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 80 * t)
    data = signal[np.newaxis, :]

    filtered = _lowpass_array(data, sfreq, cutoff=20.0)

    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    idx_2hz = np.argmin(np.abs(freqs - 2.0))
    idx_80hz = np.argmin(np.abs(freqs - 80.0))

    psd_in = np.abs(np.fft.rfft(data[0])) ** 2
    psd_out = np.abs(np.fft.rfft(filtered[0])) ** 2

    # 2 Hz: preserved (ratio > 0.5)
    assert psd_out[idx_2hz] / psd_in[idx_2hz] > 0.5
    # 80 Hz: strongly attenuated (ratio < 0.01)
    assert psd_out[idx_80hz] / psd_in[idx_80hz] < 0.01


# %% _src_to_deriv


@pytest.mark.parametrize(
    "src, suffix, expected",
    [
        ("head_eul_x", "_vel", "head_eul_vel_x"),
        ("head_eul_y", "_acc", "head_eul_acc_y"),
        ("head_pos_z", "_vel", "head_pos_vel_z"),
        ("hand_vel_x", "_acc", "hand_vel_acc_x"),
    ],
)
def test_src_to_deriv(src, suffix, expected):
    """Build correct derivative channel name."""
    assert _src_to_deriv(src, suffix) == expected


def test_src_to_deriv_fallback():
    """Append suffix to unrecognised channel names."""
    result = _src_to_deriv("mystreamchannel", "_vel")
    assert result == "mystreamchannel_vel"
