"""Project-specific analyses.

Each subpackage corresponds to one research project and may contain modules
for any combination of ERP, decoding, source, group-level, or statistical
analyses specific to that project.

Subpackages
-----------
mne_denoise : MNE denoising comparison project (HIP dataset).
"""

from bpn_analysis.analysis import mne_denoise

__all__ = ["mne_denoise"]
