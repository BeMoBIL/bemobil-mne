"""Rigid body kinematics processing with motion.py utilities.

Demonstrates:
- Building a synthetic MNE Raw object with quaternion and position channels
  following the bpn_analysis naming convention.
- find_rigid_bodies(raw) - detect rigid body names from channel labels.
- process_rigid_body(raw, ...) - convert quaternions to Euler angles, compute
  velocity and acceleration.
- split_by_rigid_body(raw) - split into per-body Raw sub-objects.

The synthetic data is runnable without any real recording.
Channels follow the convention: <body>_quat_[xyzw], <body>_pos_[xyz].
"""

# %% Imports

import mne
import numpy as np

from bpn_analysis.preproc import (
    find_rigid_bodies,
    process_rigid_body,
    split_by_rigid_body,
)

# %% Settings & Constants

SFREQ = 120.0     # typical motion capture rate (Hz)
DURATION = 10.0   # seconds

# %% Build synthetic Raw with motion capture channels

rng = np.random.default_rng(42)
n_times = int(SFREQ * DURATION)

# Two rigid bodies: "head" and "rhand" (right hand)
# Quaternion channels: _quat_x, _quat_y, _quat_z, _quat_w
# Position channels:   _pos_x, _pos_y, _pos_z
ch_names = [
    "head_quat_x", "head_quat_y", "head_quat_z", "head_quat_w",
    "head_pos_x",  "head_pos_y",  "head_pos_z",
    "rhand_quat_x", "rhand_quat_y", "rhand_quat_z", "rhand_quat_w",
    "rhand_pos_x",  "rhand_pos_y",  "rhand_pos_z",
]
n_ch = len(ch_names)

data = rng.standard_normal((n_ch, n_times)) * 0.01

# Unit-normalise the quaternion rows so they behave like real orientation data
for quat_start in (0, 7):
    q = data[quat_start : quat_start + 4]
    data[quat_start : quat_start + 4] = q / np.linalg.norm(q, axis=0, keepdims=True)

info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=["misc"] * n_ch)
raw = mne.io.RawArray(data, info, verbose=False)

print("Channels in raw:", raw.ch_names)

# %% Detect rigid bodies

rb_names = find_rigid_bodies(raw)
print("Detected rigid bodies:", rb_names)  # ['head', 'rhand']

# %% Process rigid bodies

raw_motion, rb_info = process_rigid_body(
    raw,
    lowpass_orient=8.0,        # low-pass orientation signals at 8 Hz
    lowpass_pos=8.0,           # low-pass position at 8 Hz
    lowpass_deriv=24.0,        # low-pass velocity/acceleration at 24 Hz
    compute_velocity=True,
    compute_acceleration=True,
)

# Quaternion channels are replaced by Euler angles; velocity and acceleration
# channels are appended.
print("\nChannels after processing:")
for ch in raw_motion.ch_names:
    print(" ", ch)

# rb_info contains a processing summary per body
for rb, info_entry in rb_info.items():
    print(f"\n{rb}: mode={info_entry['mode']}, channels={info_entry['channels']}")

# %% Split by rigid body

body_raws = split_by_rigid_body(raw_motion)

for name, body_raw in body_raws.items():
    print(f"\n{name}: {body_raw.n_times} samples, {len(body_raw.ch_names)} channels")
    print("  ", body_raw.ch_names)
