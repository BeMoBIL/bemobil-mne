# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Make the package importable from the repo root
sys.path.insert(0, str(Path(__file__).parents[2]))

# -- Project information -----------------------------------------------------

project = "BPN Analysis"
copyright = "2026, Roy Eric Wieske"
author = "Roy Eric Wieske"

try:
    from bpn_analysis import __version__
    release = __version__
except Exception:
    release = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # pulls docstrings from the source
    "sphinx.ext.autosummary",   # generates summary tables
    "sphinx.ext.intersphinx",   # cross-links to MNE, NumPy, etc.
    "sphinx.ext.viewcode",      # adds [source] links on every object page
    "numpydoc",                 # renders NumPy-style Parameters/Returns sections
    "sphinx_copybutton",        # copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_typehints = "description"   # put type hints in the description, not signature
autosummary_generate = True         # auto-create stub .rst files

# -- numpydoc ----------------------------------------------------------------

numpydoc_show_class_members = False     # autosummary handles class members
numpydoc_xref_param_type = True         # turn type names into cross-refs

# -- intersphinx -------------------------------------------------------------
# Lets :class:`mne.io.Raw`, :func:`numpy.array`, etc. resolve to live links.

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "mne": ("https://mne.tools/stable", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/Randomidous/data-analysis",
    "show_toc_level": 2,
}
