"""PyBeMoBIL: MoBI data loading, preprocessing, and visualization."""

from importlib.metadata import PackageNotFoundError, version

from pybemobil import io, preproc, viz
from pybemobil.io import XDFLoader

try:
    __version__ = version("pybemobil")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["XDFLoader", "__version__", "io", "preproc", "viz"]
