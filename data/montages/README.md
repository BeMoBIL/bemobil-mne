# Note regarding montages

Matlab and Python assume different coordinate systems for EEG electrode positions. The original `standard_MoBI_128.elc` file is in the Matlab coordinate system, which can lead to incorrect electrode placements when used in Python.

When using the montage in Python, please ensure that you use the corrected version to avoid any discrepancies in electrode placements.

Whenever you work with a montage the first time, load and plot it (montage.plot()) to verify that the electrode positions are correct. If you notice any discrepancies, please correct the coordinates and save the corrected montage.
