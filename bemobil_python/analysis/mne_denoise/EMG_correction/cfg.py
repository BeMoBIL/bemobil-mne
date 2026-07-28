"""Configuration and settings for the EMG mne-denoise pipeline."""

from pathlib import Path

from bpn_analysis import XDFLoader
from bpn_analysis.preproc import EEGPreprocessor

# %% run mode

SKIP_THESE = []

MODE = True  # can be "single", "group", or True (both)

FORCE_RERUN = True  # set True to reprocess subjects even if all outputs exist


# %% paths

DATA_DIR = Path(
    r"C:\Users\random\OneDrive - Zander Labs\Desktop\datasets\spot_rotation"
)
BIDS_ROOT = DATA_DIR  # BIDS dataset root
DERIV_DIR = DATA_DIR / "derivatives"

PIPELINE_NAME = "EMG_correction"

README_DESTINATION = DERIV_DIR / PIPELINE_NAME / "README.md"

# %% BIDS reading settings

SESSION = "body"  # only ses-body folders are processed
TASK = "Rotation"
TARGET_SFREQ = 500.0

# %% neck channel settings

# Channels whose names start with this prefix are treated as neck EMG.
NECK_CHANNEL_PREFIX = "BrainVision RDA_N"

# N29–N31 were never physically placed — absent from electrodes.tsv across all
# 40 recordings. Dropped on load to avoid flat channels entering preprocessing.
CHANNELS_ABSENT: frozenset[str] = frozenset(
    {"BrainVision RDA_N29", "BrainVision RDA_N30", "BrainVision RDA_N31"}
)

# N32 is the FCz scalp reference (z ≈ 151 mm); excluded from the neck array
# but kept in the raw (valid scalp electrode).
NECK_CHANNEL_EXCLUDE: frozenset[str] = frozenset({"BrainVision RDA_N32"})

# kept for backward-compat; overridden at runtime by _detect_neck_channels()
NECK_CHANNEL_NAMES: list[str] = []

# Number of PCA components to compute from neck channels; kurtosis selects the best one.
N_NECK_PCA_COMPONENTS = 10

# Minimum absolute Pearson |r| between a source and the neck PCA reference
# required to mark that IC for exclusion in the ICA+neck method.
# If no source exceeds this, the single most-correlated IC is excluded.
NECK_CORR_THRESHOLD = 0.30

# %% spectral / epoch settings

EMG_BAND = (30.0, 100.0)  # Hz – used for burst-locked amplitude metric
ALPHA_BETA_BAND = (8.0, 30.0)  # Hz – used for low-frequency preservation metric

EPOCH_TIMES = (-0.2, 0.8)
BASELINE = (-0.2, 0)

BANDPASS_ERP = (None, 20.0)

# stimulus onset delay
TSHIFT = 0.060  # seconds

# %% group / responder settings

RESPONDER_THRESHOLD_PCT = 50.0  # % RMS reduction to classify as responder

# %% dipolarity threshold

RV_THRESHOLD = 0.15  # residual variance < 15% → dipolar component

# %% BIDS helpers

# maps correction method keys → BIDS desc labels
_EMG_DESC = {
    "linear_dss": "linearDSS",
    "nonlinear_dss": "nonlinearDSS",
    "ica_head": "icaHead",
    "ica_neck": "icaNeck",
}

_BIDS_ENTITY_ORDER = ("sub", "ses", "task", "acq", "run")

# %% loader / preprocessor

# XDFLoader is retained solely to satisfy EEGPreprocessor's constructor signature;
# data loading is now handled via mne-bids (read_raw_bids).
LOADER = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=[],
    target_sfreq=TARGET_SFREQ,
)

# Preprocessing ICA excludes eye/line/channel noise but NOT muscle artifacts —
# those are the target of the four EMG correction methods below.
PREPROCESSOR = EEGPreprocessor(
    LOADER,
    asr=False,
    exclude_labels=["eye blink", "line noise", "channel noise"],
    include_labels=None,
    rng_seed=1836791205,
)
