# Note regarding montages

Matlab and Python assume different coordinate systems for EEG electrode positions. The original `standard_MoBI_128.elc` file is in the Matlab coordinate system, which can lead to incorrect electrode placements when used in Python.

When using the montage in Python, please ensure that you use the corrected version (`standard_MoBI_128_corrected.fif`) to avoid any discrepancies in electrode placements.

The files are installed with the package. To use the corrected montage:

```python
from importlib.resources import files

from bemobil_mne.io import XDFLoader

montage_dir = files("bemobil_mne") / "data" / "montages"
loader = XDFLoader(montage=montage_dir / "standard_MoBI_128_corrected.fif")
# or: mne.channels.read_dig_fif(montage_dir / "standard_MoBI_128_corrected.fif")
```

Whenever you work with a montage the first time, load and plot it (montage.plot()) to verify that the electrode positions are correct. If you notice any discrepancies, please correct the coordinates and save the corrected montage.
