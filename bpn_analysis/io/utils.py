"""Holds helpers for I/O operations in the BPN-analysis."""

# %% Imports

import json

import numpy as np


# %% Constants & Settings


# %% Classes & Functions


class NumpyEncoder(json.JSONEncoder):
    """Help encoding numpy arrays in JSON."""

    def default(self, obj):
        """Convert numpy arrays and floats to lists and native floats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


# %%
