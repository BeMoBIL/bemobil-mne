# data-analysis

Python repository for EEG and multimodal analysis utilities at BPN (Berlin Mobile Brain/Body Imaging Lab).

## Structure

```
bpn_analysis/
├── io/                          # Data I/O
│   ├── xdf.py                   # XDFLoader: multimodal XDF loading, stream alignment
│   ├── alignment.py             # align_stream_to_timestamps (timestamp-aware interpolation)
│   ├── bids_export.py           # BIDS export wrappers around mne_bids
│   └── utils.py                 # I/O helpers
├── preproc/                     # Preprocessing
│   ├── preprocessing.py         # EEGPreprocessor: filter -> bad ch -> ASR -> ICA pipeline
│   ├── epoching.py              # EpochPreparer, stimulus renaming helpers
│   ├── motion.py                # Rigid body kinematics (find, process, split)
│   └── utils.py                 # ASR, ZapLine, ICA, dipole fitting, provenance, misc
├── viz/                         # Visualization
│   ├── plotting.py              # ERP, PSD, and component plots
│   └── utils.py                 # Plot helpers
├── analysis/                    # Project-specific analysis scripts
│   ├── mne_denoise/             # MNE-denoise pipeline comparisons
│   └── neuro_urban_walks/       # Neuro-urban walking study scripts
├── examples/                    # Runnable example scripts
│   ├── preprocess_and_plot.py   # Full EEG preprocessing + visualization walkthrough
│   ├── xdf_loading.py           # XDF multimodal loading
│   ├── motion_processing.py     # Rigid body kinematics
│   └── bids_export_example.py   # BIDS export
└── data/
    └── montages/                # EEG electrode montage files (.elc, .fif)
```

## Modules

**`io`** - Load XDF multimodal recordings into MNE Raw objects, align auxiliary streams (ECG, gaze, EMG) to a common time grid, and export datasets to BIDS.

**`preproc`** - Full EEG preprocessing pipeline (bandpass, bad channel detection, ASR, ICA, dipole fitting), epoch preparation, and rigid body motion capture kinematics.

**`viz`** - ERP and PSD plotting utilities built on MNE and matplotlib.

**`analysis`** - Project-specific scripts. Not part of the importable API.

**`examples`** - Self-contained scripts demonstrating major features. Run directly or open as notebooks (percent-format cells).

## Quick start

```bash
# Navigate to the repo root, then install in editable mode
pip install -e .
```

```python
from bpn_analysis.io import XDFLoader
from bpn_analysis.preproc import EEGPreprocessor, find_rigid_bodies

# Load a multimodal XDF recording
loader = XDFLoader(montage="standard_1020")
recording = loader.load("sub-01_task-walk_run-01.xdf")
raw = recording.raw
```

See `bpn_analysis/examples/` for complete worked examples:

| Feature | Example script |
|---------|---------------|
| XDF multimodal loading | `examples/xdf_loading.py` |
| EEG preprocessing pipeline | `examples/preprocess_and_plot.py` |
| Rigid body kinematics | `examples/motion_processing.py` |
| BIDS export | `examples/bids_export_example.py` |

## Contributing

This is a private repository for BPN lab members. Contributions are made via feature branches and pull requests rather than forks.

### Setup

1. Clone the repo:
```bash
git clone https://github.com/Randomidous/data-analysis.git
cd data-analysis
```
2. Install in editable mode:
```bash
pip install -e .
```
3. Install pre-commit hooks:
```bash
pre-commit install
```

### Workflow

1. Create a branch from `main`:
```bash
git checkout -b your-git-name/short-description
```
2. Make your changes, commit often with clear messages
3. Push your branch:
```bash
git push -u origin your-git-name/short-description
```
4. Open a pull request against `main` on GitHub
5. Request a review from a lab member before merging

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `name/feature-desc` | `robin/add-ica-pipeline` |
| Fix | `name/fix-desc` | `alex/fix-montage-rotation` |
| Experiment | `name/exp-desc` | `maria/exp-zapline-threshold` |

### Guidelines

- Keep PRs focused - one logical change per PR
- Pre-commit hooks run automatically on commit (formatting, linting)
- Do not push directly to `main`, always use branches and keep main clean (mirror of upstream)
- Include a short description in your PR of what changed and why
