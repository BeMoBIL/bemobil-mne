"""Preprocess data for analysis of the Neuro Urban Walks dataset.

Recording setup
---------------
- EEG:          LiveAmp (Brain Products), type "EEG"
- ECG:          separate amplifier channel, type "ECG"
- EDA:          separate amplifier channel, type "EDA"
- Eye-tracking: Pupil Labs Core/Neon via LSL plugin, type "Gaze"
                Fixation/saccade/blink streams are irregular and go to events dict
- Audio:        ambient/microphone recording, type "Audio" (Tier-2, native 44.1 kHz)
- Markers:      LSL Markers stream for walk-segment triggers

Stream type strings below must match what was configured in LSL at recording time.
Verify with pyxdf.load_xdf() on a pilot file if any stream is not found at runtime.
"""

# %% Imports

from pathlib import Path

import numpy as np

from bpn_analysis import XDFLoader
from bpn_analysis.io import MultimodalRecording
from bpn_analysis.preproc.preprocessing import EEGPreprocessor

# %% Constants & Settings

DATA_DIR = Path(r"\\bpn-data.bpn.tu-berlin.de\projects\Project_NeuroUrbWalks\data")
DERIV_DIR = DATA_DIR / "derivatives"
FILES = list(DATA_DIR.rglob("*.xdf"))

TARGET_SFREQ = 500.0

# %% Loader

LOADER = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=["x_dir", "y_dir", "z_dir"],
    target_sfreq=TARGET_SFREQ,
    special_streams={
        # ECG from separate amplifier (~1000 Hz -> 500 Hz: sinc to avoid aliasing)
        "ecg": {"type": "ECG", "method": "sinc"},
        # Pupil Labs gaze: x/y/confidence/pupil_diameter (~200 Hz -> 500 Hz: pchip)
        "gaze": {"type": "Gaze", "method": "pchip"},
        # EDA from separate amplifier
        "eda": {"type": "EDA", "method": "sinc"},
    },
    tier2_streams={
        # Audio at 44.1 kHz, preserved at native rate
        "audio": {"type": "Audio"},
    },
    marker_stream_types=["Markers", "Logging", "Notes"],
    alignment_method="pchip",  # fallback for any unlisted special stream
    max_nan_gap_s=0.5,
    on_mismatch="pad",
)

# %% Preprocessor
#
# EEGPreprocessor.run_raw() is called on rec.raw (Tier-1 only).
# We do not use EEGPreprocessor.run() because it calls loader.load() internally
# and expects bare mne.io.Raw; our loader returns MultimodalRecording.

PREPROCESSOR = EEGPreprocessor(
    loader=LOADER,  # stored but run_raw() is used below
    filter_bands=(0.1, 100.0),
    filter_bands_ica=(1.0, 100.0),
    notch_freqs=(50, 100, 150),
    downsample_ica=250.0,
    thresh=0.7,
    asr_cutoff=20.0,
    rng_seed=42,
    include_labels=frozenset({"brain", "other"}),
    # ECG and gaze are merged as misc/ecg channels; set types so ICA ignores them
    channel_types={"ecg": "ecg", "eda": "eda"},
)

# %% Main driver


def main():
    """Run main."""
    for fname_in in FILES:
        print(f"Processing {fname_in}...")

        # -- Load all streams ----------------------------------------------
        rec: MultimodalRecording = LOADER.load(fname_in)

        # -- Preprocess Tier-1 (EEG + ECG + gaze) -------------------------
        fname_out = (
            DERIV_DIR / fname_in.relative_to(DATA_DIR).with_suffix("") / fname_in.stem
        )

        PREPROCESSOR.run_raw(rec.raw, fname_out=fname_out, overwrite=True)

        # -- Save Tier-2: audio at native rate -----------------------------
        if "audio" in rec.tier2:
            audio_data, audio_ts = rec.tier2["audio"]
            np.savez_compressed(
                fname_out.with_suffix(".audio.npz"),
                data=audio_data,
                timestamps_s=audio_ts,
                session_t0=rec.session_t0,
            )
            print(
                f"  Audio: {audio_data.shape[0]} samples, "
                f"{audio_ts[-1]:.1f} s, "
                f"~{audio_data.shape[0] / audio_ts[-1]:.0f} Hz"
            )

        # -- Log irregular event streams (fixations, saccades, blinks) ----
        for ev_label, (descriptions, ev_ts) in rec.events.items():
            print(f"  Events '{ev_label}': {len(ev_ts)} events")

        print(f"  session_t0 = {rec.session_t0:.3f} s (LSL)")


# %% Entry point

if __name__ == "__main__":
    main()
