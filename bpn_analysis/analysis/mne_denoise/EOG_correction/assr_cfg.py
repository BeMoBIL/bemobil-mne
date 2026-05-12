"""Configuration and settings for the ASSR mne-denoise pipeline."""

from pathlib import Path

from bpn_analysis import XDFLoader
from bpn_analysis.preproc import EEGPreprocessor

# %% run mode

SKIP_THESE = ["13", "17", "16"]

MODE = "group"  # can be "single", "group", or True (both)

FORCE_RERUN = True  # set True to reprocess subjects even if all outputs exist

RESPONDER_THRESHOLD_PCT = 50.0  # minimum % RMS reduction to classify as responder

# %% paths

DATA_DIR = Path(r"C:\Users\random\OneDrive - Zander Labs\Desktop\datasets\assr")
DERIV_DIR = DATA_DIR / "derivatives"

PIPELINE_NAME = "EOG_correction"

README_DESTINATION = DERIV_DIR / PIPELINE_NAME / "README.md"


# %% epoch / analysis settings

REMAPS = {}

EPOCH_TIMES = (-0.2, 0.8)
BASELINE = (-0.2, 0)

BANDPASS_ERP = (None, 20.0)

# stimulus onset delay: shift trigger annotations forward by this amount so
# that t=0 in the epoch aligns with actual stimulus presentation.
# 60 ms is a typical LSL/screen-refresh latency for this setup.
TSHIFT = 0.060  # seconds

# %% BIDS helpers

# maps eog_cleaned dict keys → BIDS desc labels
_EOG_DESC = {
    "linear_dss": "linearDSS",
    "nonlinear_dss": "nonlinearDSS",
    "ica": "icaEOG",
}

_BIDS_ENTITY_ORDER = ("sub", "ses", "task", "acq", "run")

# %% loader / preprocessor

LOADER = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=[],  # ["x_dir", "y_dir", "z_dir"],
    target_sfreq=500.0,
)

PREPROCESSOR = EEGPreprocessor(
    LOADER,
    rng_seed=1836791205,
    asr_cutoff=20,
    exclude_labels=["muscle artifact", "line noise", "channel noise"],
    include_labels=None,
)
