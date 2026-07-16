"""Holds utilities for visualization in the BPN-analysis."""

# %% Imports

import gc

import matplotlib._pylab_helpers as plab_helpers
import matplotlib.pyplot as plt

# %% Constants & Settings


# %% Classes & Functions


def clear_matplotlib_memory():
    """Clear and close all matplotlib figures and caches."""
    plt.close("all")
    plab_helpers.Gcf.destroy_all()
    gc.collect()
