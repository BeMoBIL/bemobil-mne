"""Show how to use the `viz` module.

Run ``preprocessing.py`` first to generate the files this example loads.
"""

# %%
# Imports

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import bemobil_mne
from bemobil_mne.viz import (
    glue_imgs,
    plot_ERP,
    plot_evoked_data,
    plot_joint,
    plot_PSD,
    plot_psd_topomaps,
    plot_TFR,
    plot_tfr_topomaps,
    plot_topo,
    plot_various_ERPs,
)

# %%
# Load preprocessed files

# These files are produced by preprocessing.py.
save_path = (
    Path(bemobil_mne.__file__).parent.parent / "reports" / "preprocess_plot_example"
)
fname_out = save_path / "testfile"

raw_minimal = mne.io.read_raw(f"{fname_out}_minimal.fif.gz", verbose="error")
raw_clean = mne.io.read_raw(f"{fname_out}_clean.fif.gz", verbose="error", preload=True)
ica = mne.preprocessing.read_ica(f"{fname_out}_ica.fif.gz", verbose="error")

with open(f"{fname_out}_iclabels.json") as fin:
    ic_labels = json.load(fin)

with open(f"{fname_out}_bad_channels.json") as fin:
    bad_ch_dict = json.load(fin)

# %%
# Set keyword arguments for plotting functions

# An easy way to pass arguments at many points is to define keyword arguments.
epochs_kwargs = {"tmin": -0.2, "tmax": 1, "baseline": (None, 0)}

# keyword arguments for plotting
# can be any parameter accepted by the functions listed below
plot_kwargs = {
    "colors": None,
    "linestyles": None,
    "styles": None,
    "vlines": "auto",
    "truncate_yaxis": "auto",
    "truncate_xaxis": True,
    "ylim": None,
    "invert_y": False,
    "title": None,
    "sphere": None,
    "time_unit": "s",
}

# %%
# Prepare input for plotting functions

# Pick the events you want to contrast
event_list = ["T0", "T2"]
events = mne.events_from_annotations(raw_minimal)[0]
event_ids = mne.events_from_annotations(raw_minimal)[1]
event_id_plotting = {event: event_ids[event] for event in event_list}

# Pick any channels you want
picks_eeg = raw_clean.ch_names

# This is the standard plotting input: a dict of Epochs keyed by condition
epoch_dict = {}
for event, idx in event_id_plotting.items():
    epochs = mne.Epochs(
        raw_clean,
        events,
        event_id={event: idx},
        picks=picks_eeg,
        preload=True,
        **epochs_kwargs,
    )
    epoch_dict[event] = epochs

# %%
# Plot ICA sources ERP

sources = ica.get_sources(raw_clean)
picks_sources = range(1, 7)

# If we don't set channel types from "misc" back to "eeg", weird scalings happen.
# However, the units will still be off.
sources.set_channel_types({ch: "eeg" for ch in sources.ch_names})
epoch_sources_dict = {}
for event, idx in event_id_plotting.items():
    epochs = mne.Epochs(
        sources,
        events,
        event_id={event: idx},
        preload=True,
        **epochs_kwargs,
    )
    epoch_sources_dict[event] = epochs


# %% Event-related potentials (ERPs)

# Raw data ERPs
fig = plot_ERP(
    epoch_dict,
    ci=0.68,
    picks=None,
    combine=None,  # set "combine" to "mean" to get the average of "picks"
    is_sources=False,
    kwargs={},
)
plt.show()

# ICA sources ERPs
fig = plot_ERP(
    epoch_sources_dict,
    ci=0.68,
    picks=None,
    combine=None,
    is_sources=True,
    kwargs={},
)
plt.show()

# `fig` is a matplotlib figure; save it with:
#  >>> fig.savefig(..., bbox_inches="tight")

# %% Joint plots, topomaps, and butterfly plots via plot_evoked_data

# `plot_evoked_data` is a higher-level wrapper that calls plot_joint, plot_topo,
# and plot_butterfly in one shot.

joint_figs, topo_figs, butter_figs = plot_evoked_data(
    epoch_dict,
    kinds=["joint", "topo", "butterfly"],
)
plt.show()

# %% Joint plots (standalone)

# These are good visualizations for an overview of all channel ERPs together
# with global field power (GFP) and topographical plots.

figs = plot_joint(epoch_dict)

# figs is a dict with event names as keys and matplotlib figures as values.
savenames = []
for key, fig in figs.items():
    savename = fname_out.parent / f"{fname_out.stem}_joint-{key.replace('/', '-')}.png"
    fig.savefig(savename, dpi=300, bbox_inches="tight")
    savenames.append(savename)

# With `glue_imgs`, combine the images into one (useful for reports)
glue_imgs(savenames, fname_out.parent / f"{fname_out.stem}_joint-COMBINED.png")

# %% Topographical plots

figs = plot_topo(epoch_dict)

savenames = []
for key, fig in figs.items():
    savename = fname_out.parent / f"{fname_out.stem}_topo-{key.replace('/', '-')}.png"
    fig.savefig(savename, dpi=300, bbox_inches="tight")
    savenames.append(savename)

glue_imgs(savenames, fname_out.parent / f"{fname_out.stem}_topo-COMBINED.png")

# %% Plot other ERPs

# If you are interested in:
#   - group average ERPs
#   - individual ERPs
#   - difference waveforms
# `plot_various_ERPs` is the function to use.

# Individual plotting (pass a single raw)
fig = plot_various_ERPs(
    raw_minimal,
    event_id=event_id_plotting,
    picks=picks_eeg,
    plot_kwargs=plot_kwargs,
    epochs_kwargs=epochs_kwargs,
)

# Group plotting (pass a list of raws)
fig = plot_various_ERPs(
    [raw_minimal, raw_clean],
    event_id=event_id_plotting,
    picks=picks_eeg,
    plot_kwargs=plot_kwargs,
    epochs_kwargs=epochs_kwargs,
)

# Difference waveforms (subtraction of two or more events)
fig = plot_various_ERPs(
    raw_clean,
    event_id=event_id_plotting,
    picks=picks_eeg,
    subtraction_event="T0",
    subtraction_targets=["T2"],
    plot_kwargs=plot_kwargs,
    epochs_kwargs=epochs_kwargs,
)

# %% TFR (a.k.a. ERSP, ERDS, etc.)

# Note: `plot_ERSP` is deprecated in bemobil_mne; use `plot_TFR` instead.

# For raw EEG data
figs = plot_TFR(
    epoch_dict,
    freqs=np.arange(5, 20),
    tmin=-0.2,
    tmax=1,
    baseline=(None, 0),
    picks=picks_eeg,
    combine=None,
)

# For ICA sources
figs = plot_TFR(
    epoch_sources_dict,
    freqs=np.arange(1, 41),
    tmin=-0.5,
    tmax=1,
    baseline=(None, 0),
    combine=None,
)

# figs is a dict with event names as keys and matplotlib figures as values.
savenames = []
for key, fig in figs.items():
    savename = fname_out.parent / f"{fname_out.stem}_tfr-{key.replace('/', '-')}.png"
    fig.savefig(savename, dpi=300, bbox_inches="tight")
    savenames.append(savename)

glue_imgs(savenames, fname_out.parent / f"{fname_out.stem}_tfr-COMBINED.png")

# %% Power Spectral Density (PSD)

figs = plot_PSD(
    epoch_dict,
    fmin=2,
    fmax=40,
    picks=picks_eeg,
    combine=False,
)

# %% Plot PSD topomaps

# For topography plots we want all channels.
# We create our standard epoch dict again with picks="all".
epoch_topo_dict = {}
for event, idx in event_id_plotting.items():
    epochs = mne.Epochs(
        raw_clean,
        events,
        event_id={event: idx},
        picks="all",
        preload=True,
        **epochs_kwargs,
    )
    epoch_topo_dict[event] = epochs

# Plot topomaps of the PSD
fig = plot_psd_topomaps(epoch_topo_dict, unit="relative", bands=None, plot_kwargs=None)

# %% Plot TFR topomaps over time

tfr_dict = {}
for event, epochs in epoch_topo_dict.items():
    freqs = np.arange(1, 50, 1)
    tfr = epochs.compute_tfr(
        freqs=freqs,
        n_cycles=freqs / 2,
        return_itc=False,
        average=False,
        method="multitaper",
    )
    tfr_dict[event] = tfr

bands = {
    "Delta\n (1-4 Hz)": (1, 4),
    "Theta\n (4-8 Hz)": (4, 8),
    "Alpha\n (8-12 Hz)": (8, 12),
    "Beta\n (12-30 Hz)": (12, 30),
    "Gamma\n (30-45 Hz)": (30, 45),
}
times = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

# Use precomputed TFRs
figs = plot_tfr_topomaps(
    tfr_dict,
    bands=bands,
)

# Alternatively, pass the epochs directly and let `plot_tfr_topomaps`
# compute the TFRs internally (equivalent to the precomputed call above,
# just slower because it recomputes the multitaper TFR over all channels):
#
#   figs = plot_tfr_topomaps(epoch_topo_dict, bands=bands)
# %%
