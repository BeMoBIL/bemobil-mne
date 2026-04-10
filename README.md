# data-analysis

Python repository for analysis utilities and project-specific scripts at BPN (Berlin Mobile Brain/Body Imaging Lab).

## Structure

bpn_analysis/
├── analysis/
│   ├── mne_denoise/
│   │   ├── denoise_comparison.py
│   │   └── hip_preprocess.py
│   ├── other_projects/
│   │   ├── project-specific-script-1.py
│   │   └── project-specific-script-2.py
│   ├── io/                         # Data In/Out module
│   │   ├── xdf.py                  # Functions for reading XDF files
│   │   └── other-io-utils.py       # Other I/O functions
│   ├── preproc/                    # Preprocessing module
│   │   ├── epoching.py             # Epoching functions
│   │   └── preprocessing.py        # Preprocessing functions
│   └── viz/                        # Visualization module
│       └── plotting.py             # Plotting functions
└── data/
    ├── montages/                   # EEG electrode montage files
    │   ├── standard_MoBI_128.elc
    │   └── standard_MoBI_128_corrected.fif
    └── other-data/


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



