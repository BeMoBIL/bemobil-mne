"""BPN-analysis: EEG data analysis scripts."""

from bpn_analysis import analysis, io, preproc, viz
from bpn_analysis.io import XDFLoader

__all__ = ["XDFLoader", "analysis", "io", "preproc", "viz"]
