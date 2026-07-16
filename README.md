# data-analysis

Python repository for analysis utilities and project-specific scripts at BPN (Berlin Mobile Brain/Body Imaging Lab).

## Structure

```
bpn_analysis/
├── analysis/
│   ├── mne_denoise/
│   │   ├── denoise_comparison.py
│   │   └── hip_preprocess.py
│   └── other_projects/
│       ├── project-specific-script-1.py
│       └── project-specific-script-2.py
├── io/                       # Data In/Out module
│   ├── xdf.py                # Functions for reading XDF files
│   └── other-io-utils.py     # Other I/O functions
├── preproc/                  # Preprocessing module
│   ├── epoching.py           # Epoching functions
│   └── preprocessing.py      # Preprocessing functions
├── viz/                      # Visualization module
│   └── plotting.py           # Plotting functions
└── data/                     # Values that convey information
    ├── montages/             # EEG electrode montage files
    │   ├── standard_MoBI_128.elc
    │   └── standard_MoBI_128_corrected.fif
    └── other-data/
```


## Requirements

Managed via `pyproject.toml`. Install with:
navigate to project folder

```bash
pip install -e .
```
Note: The `-e` flag installs the package in editable mode. This means whenever you make changes to the code in the `bpn_analysis` directory, those changes will be reflected immediately without needing to reinstall the package.

## Usage

Then, you can import the package in your Python scripts or interactive sessions:
```python
import bpn_analysis
```

or
```python
from bpn_analysis import plot_ERP, plot_PSD
```


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
