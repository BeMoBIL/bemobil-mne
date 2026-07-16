"""Preprocess data for analysis of the Neuro Urban Walks dataset."""

# %% Imports

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from bpn_analysis import XDFLoader

# %% Constants & Settings

DATA_DIR = Path(r"\\bpn-data.bpn.tu-berlin.de\projects\Project_NeuroUrbWalks\data")
DERIV_DIR = DATA_DIR / "derivatives"

EPOCH_TLIMES = (-0.2, 0.8)
BASELINE = (-0.2, 0)

BANDPASS_ERP = (None, 20.0)

LOADER = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=["x_dir", "y_dir", "z_dir"],
    target_sfreq=500.0,
)

# %% Helper functions


# %% Main driver

def main():
    """Run main."""
    pass

# %% Main

if __name__ == "__main__":
    main()
