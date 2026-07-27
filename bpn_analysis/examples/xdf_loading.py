"""XDF multimodal loading with XDFLoader.

Demonstrates:
- Configuring XDFLoader with special_streams for ECG (sinc), gaze (pchip),
  and a trigger channel (stim).
- Loading an XDF file and unpacking the MultimodalRecording result.
- Accessing raw, tier2, events, and session_t0.
- Calling align_stream_to_timestamps directly for custom alignment.

NOTE: This script requires a real XDF file. The loader.load() call is shown
but commented out. Replace XDF_PATH with your actual file path before running.
"""

# %% Imports

import numpy as np

from bpn_analysis.io import MultimodalRecording, XDFLoader, align_stream_to_timestamps

# %% Settings & Constants

XDF_PATH = "/path/to/your/recording.xdf"  # <-- set this before running

# %% Configure loader

loader = XDFLoader(
    eeg_stream_name="BrainVision RDA",
    eeg_stream_type="EEG",
    montage="standard_1020",
    old_reference="FCz",
    special_streams={
        # ECG amplifier at ~1000 Hz - use sinc (anti-aliasing + PCHIP)
        "ecg": {"name": "ECG", "type": "ECG", "method": "sinc"},
        # Pupil Labs gaze at ~200 Hz - use pchip for smooth physiological signal
        "gaze": {"name": "pupil_capture", "type": "Gaze", "method": "pchip"},
        # Button box / trigger channel - use stim to preserve sparse pulses
        "stim": {"name": "TriggersFromStim", "type": "Markers", "method": "stim"},
    },
    alignment_method="pchip",  # default for any unlisted special streams
    on_mismatch="crop",
)

# %% Load XDF file
#
# Uncomment the line below once XDF_PATH is set.
# recording = loader.load(XDF_PATH)
#
# The returned MultimodalRecording has four fields:
#
#   recording.raw        - mne.io.RawArray, Tier-1 streams aligned to a common grid
#   recording.tier2      - dict: {label: (data, timestamps_s)} at native rate
#   recording.events     - dict: {label: (descs, timestamps_s)} for irregular streams
#   recording.session_t0 - float, absolute LSL timestamp of t=0 in raw
#
# Example access pattern (shown on a placeholder object):

recording = MultimodalRecording(
    raw=None,  # would be mne.io.RawArray
    tier2={},
    events={},
    session_t0=0.0,
)

# After a real load:
#
#   raw = recording.raw
#   raw.plot()
#
#   # Tier-2 stream (e.g. high-rate audio kept at native sfreq)
#   audio_data, audio_ts = recording.tier2["audio"]   # (n_samples, n_ch), (n_samples,)
#
#   # Irregular event stream (e.g. fixation onset events from the eye-tracker)
#   fix_labels, fix_ts = recording.events["fixations"]
#
#   # Absolute time reference - cross-check with external system logs
#   print(f"Session started at LSL t={recording.session_t0:.3f} s")

# %% Align a stream manually

# align_stream_to_timestamps can also be called directly when you have raw
# (data, timestamps) from any source and want to map it onto a target grid.
#
# Example: align a 200 Hz custom sensor to the EEG time grid
rng = np.random.default_rng(0)

src_sfreq = 200.0
tgt_sfreq = 500.0
duration = 5.0

src_ts = np.arange(0, duration, 1.0 / src_sfreq)
src_data = rng.standard_normal((len(src_ts), 3))  # (n_samples, n_channels)

tgt_ts = np.arange(0, duration, 1.0 / tgt_sfreq)

aligned = align_stream_to_timestamps(
    data=src_data,
    src_ts=src_ts,
    tgt_ts=tgt_ts,
    method="pchip",
)

print(f"Source shape : {src_data.shape}")  # (1000, 3)
print(f"Aligned shape: {aligned.shape}")  # (2500, 3)
