"""BeMoBIL-MNE: MoBI data loading, preprocessing, and visualization."""

from importlib.metadata import PackageNotFoundError, version

from bemobil_mne import io, preproc, viz
from bemobil_mne.io import XDFLoader

try:
    __version__ = version("bemobil-mne")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["XDFLoader", "__version__", "io", "preproc", "viz"]
