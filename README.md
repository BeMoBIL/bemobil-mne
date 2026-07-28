# bemobil-mne

MoBI data loading, preprocessing, and visualization built on MNE-Python.

## Structure

```md
bemobil_mne/
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
from bemobil_mne.io import XDFLoader
from bemobil_mne.preproc import EEGPreprocessor, find_rigid_bodies

# Load a multimodal XDF recording
loader = XDFLoader(montage="standard_1020")
recording = loader.load("sub-01_task-walk_run-01.xdf")
raw = recording.raw
```

See `examples/` for complete worked examples:

| Feature | Example script |
| --------- | --------------- |
| XDF multimodal loading | `examples/xdf_loading.py` |
| EEG preprocessing pipeline | `examples/preprocess_and_plot.py` |
| Rigid body kinematics | `examples/motion_processing.py` |
| BIDS export | `examples/bids_export_example.py` |

## Contributing

This is a private repository for BPN lab members. Contributions are made via feature branches and pull requests rather than forks.

### Setup

1. Clone the repo:

    ```bash
    git clone https://github.com/BeMoBIL/bemobil-mne.git
    cd bemobil-mne
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
| ------ | --------- | --------- |
| Feature | `name/feature-desc` | `robin/add-ica-pipeline` |
| Fix | `name/fix-desc` | `alex/fix-montage-rotation` |
| Experiment | `name/exp-desc` | `maria/exp-zapline-threshold` |

### Testing

Tests live in `tests/` and follow the same structure as `standard_scripts`. Run the full suite from the repo root:

```bash
pip install -e ".[test]"   # first time only
pytest                     # fast tests only (default)
pytest --runslow           # include slow integration tests
```

Coverage runs automatically and produces an HTML report in `htmlcov/`. Open `htmlcov/index.html` in a browser after the run.

**Test structure:**

`tests/conftest.py` defines shared fixtures (synthetic `tiny_raw`, `motion_raw`, and the EEGBCI `sample_raw` used by slow tests) and registers two custom markers.

| Marker                        | Meaning                                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `@pytest.mark.slowtest`       | Integration test that downloads data or fits models. Skipped by default; run with `--runslow`.   |
| `@pytest.mark.requires_data`  | Test that depends on external data files. Skipped when paths are absent.                         |

Each test module maps to one source module:

| Test file | Source module |
| ----------- | -------------- |
| `test_utils.py` | `preproc/utils.py` |
| `test_motion.py` | `preproc/motion.py` |
| `test_preprocessing.py` | `preproc/preprocessing.py` |
| `test_xdf_helpers.py` | `io/xdf.py` (private helpers) |
| `test_bids_export.py` | `io/bids_export.py` |

**Integration testing convention:**

Every significant change to a public function must be accompanied by at least one test. Fast unit tests (synthetic data, no I/O) are the default. Integration tests that touch real files or run the full pipeline are marked `@pytest.mark.slowtest` and kept in the same test module as the unit tests for that module. PRs are expected to pass `pytest` (without `--runslow`) in CI.

### Guidelines

- Keep PRs focused - one logical change per PR
- Pre-commit hooks run automatically on commit (formatting, linting)
- Do not push directly to `main`, always use branches and keep main clean (mirror of upstream)
- Include a short description in your PR of what changed and why
