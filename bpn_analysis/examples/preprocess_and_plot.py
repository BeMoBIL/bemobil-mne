"""Show how to use the `preproc` and `viz` modules."""

# %%
# Imports

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci

import bpn_analysis
from bpn_analysis.preproc import EEGPreprocessor
from bpn_analysis.preproc.utils import get_descriptor
from bpn_analysis.viz import (
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
# Load raw data to show the functionality

# `EEGPreprocessor` is the main class for preprocessing.
# It can load XDF files directly via `EEGPreprocessor.run(fname_in)`,
# or you may pass an mne raw object to `EEGPreprocessor.run_raw()`, which we do here.

# Read in an mne raw object; the montage must be set before calling run_raw()
sample_paths = eegbci.load_data([1], [6], update_path=True)
raw = mne.io.read_raw(sample_paths[0], verbose=False, preload=True)
eegbci.standardize(raw)
raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

# Crop to the first 60 s: the full recording is ~125 s, but 60 s already contains
# all event types and exercises every code path below, while keeping ICA and TFR
# computation roughly twice as fast.
raw.crop(tmax=60)

event_id = {"T0": 1, "T1": 2, "T2": 3}

# %%
# Some further settings

save_path = (
    Path(bpn_analysis.__file__).parent.parent / "reports" / "preprocess_plot_example"
)
save_path.mkdir(exist_ok=True, parents=True)
fname_out = save_path / "testfile"
seed = 869330571  # good practice: pick a random seed via secrets.randbits(31)

# %%
# Run preprocessing

# This will create files in the save_path directory; see the docstring of
# `EEGPreprocessor` for details (e.g., `print(help(EEGPreprocessor))`).
#
# `run_raw` returns a 3-tuple: (raw_clean, report, metadata)
# metadata contains: raw_minimal, raw_asr, raw_subset, ica, ic_labels,
#                    dipoles, residuals, trans, bad_ch_dict
#
# Set fit_dipoles=True to also get dipole fits and trans.
# line_noise_freq accepts "europe" (50 Hz) or "usa" (60 Hz) as shortcuts,
# or a float. Harmonics up to Nyquist are computed automatically.

preprocessor = EEGPreprocessor(
    loader=None,  # not needed when calling run_raw() directly
    line_noise_freq="usa",  # 60 Hz + harmonics up to Nyquist
    filter_bands=(0.1, 70.0),  # eegbci sfreq is 160 Hz; keep h_freq below Nyquist
    filter_bands_ica=(1.0, 70.0),
    downsample_ica=None,  # sfreq is 160 Hz; cannot upsample to default 250 Hz
    rng_seed=seed,
    event_id=event_id,  # recorded in provenance metadata
    # ica_method="amica"  # default; use "picard" to fall back to MNE's picard
)

t_start = time.perf_counter()
(
    raw_minimal,
    raw_clean,
    raw_asr,
    raw_subset,
    ica,
    ic_labels,
    dipoles,
    residuals,
    trans,
    bad_ch_dict,
) = preprocessor.run_raw(
    raw,
    fname_out=fname_out,
    overwrite=True,
)
t_stop = time.perf_counter()
print(f"Processing took {t_stop - t_start:.2f} seconds.")

# %%
# Inspect processing provenance
#
# The provenance descriptor is stored in raw.info['description'] as a JSON string.
# Use `get_descriptor` to parse it into a dict.

proc_description = get_descriptor(raw_clean)
print(proc_description)

# %%
# Run with dipole fitting

# Setting fit_dipoles=True also resolves and saves trans.
# trans=None (default) uses the MNE fsaverage template; trans="fit" runs
# automatic coregistration.
# NOTE: dipole fitting can take several minutes on a standard machine.

do_dipoles = False  # set to True to enable
if do_dipoles:
    (
        raw_minimal,
        raw_clean,
        raw_asr,
        raw_subset,
        ica,
        ic_labels,
        dipoles,
        residuals,
        trans,
        bad_ch_dict,
    ) = EEGPreprocessor(
        loader=None,
        line_noise_freq="usa",
        filter_bands=(0.1, 70.0),
        filter_bands_ica=(1.0, 70.0),
        downsample_ica=None,
        fit_dipoles=True,
        trans=None,  # use fsaverage template
        rng_seed=seed,
    ).run_raw(raw, fname_out=fname_out, overwrite=True)

# %%
# You may also run the pipeline without ASR (asr=False is the default),
# or with custom ASR settings via asr={"cutoff": 10}, and/or without ICA via fit_ica=False.
# Set `fname_out=None` to skip saving.

t_start = time.perf_counter()
(
    raw_minimal,
    raw_clean,
    raw_asr,
    raw_subset,
    ica,
    ic_labels,
    dipoles,
    residuals,
    trans,
    bad_ch_dict,
) = EEGPreprocessor(
    loader=None,
    line_noise_freq="usa",
    filter_bands=(0.1, 70.0),
    filter_bands_ica=(1.0, 70.0),
    downsample_ica=None,
    # asr=False is the default - ASR is skipped
    rng_seed=seed,
).run_raw(raw, fname_out=None)
t_stop = time.perf_counter()
print(f"Processing took {t_stop - t_start:.2f} seconds.")

# %%
# Reading in files

# Above, we assigned outputs of `run_raw` to different variables.
# If you ran the pipeline before and only have the saved files, read them back:

# Read back minimally processed data
raw_minimal = mne.io.read_raw(f"{fname_out}_minimal.fif.gz", verbose="error")

# Read back ICA cleaned data
raw_clean = mne.io.read_raw(f"{fname_out}_clean.fif.gz", verbose="error")

# Read back ASR cleaned data
raw_asr = mne.io.read_raw(f"{fname_out}_asr.fif.gz", verbose="error")

# Read back ICA solution
ica = mne.preprocessing.read_ica(f"{fname_out}_ica.fif.gz", verbose="error")

# Read back IC labels
with open(f"{fname_out}_iclabels.json") as fin:
    ic_labels = json.load(fin)

# Read back bad channels (these have been interpolated in raw_clean)
with open(f"{fname_out}_bad_channels.json") as fin:
    bad_ch_dict = json.load(fin)

# If dipoles were saved, read them back like this:
#   dipdir = Path(f"{fname_out}_dipoles")
#   dipoles = [mne.read_dipole(f) for f in sorted(dipdir.glob("ic-*-dip.bdip"))]
#   residuals = [
#       mne.read_evokeds(f, verbose="error")[0]
#       for f in sorted(dipdir.glob("ic-*-residual.fif.gz"))
#   ]
#   trans = mne.transforms.read_trans(f"{fname_out}_trans.fif", verbose="error")

# %%
# Plot a dipole corresponding to an IC (only when fit_dipoles=True was used)

# index 0 will likely correspond to an eye movement IC
ic_idx = 0

if dipoles:
    # plot the component
    fig_ica = ica.plot_components(ic_idx, show=False)

    # plot the dipole in 2D
    fig_dip = mne.viz.plot_dipole_locations(
        dipoles[ic_idx],
        subject="fsaverage",
        trans=trans,
        mode="outlines",
        show=False,
    )
    plt.show()

    # plot a topomap of the residuals
    fig_resid = residuals[ic_idx].plot_topomap(times=0, show=False)
    plt.show()

# %% Set keyword arguments for plotting functions

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

# %% Prepare input for plotting functions

# Pick the events you want to contrast
event_list = ["T0", "T2"]
events = mne.events_from_annotations(raw)[0]
event_ids = mne.events_from_annotations(raw)[1]
event_id_plotting = {event: event_ids[event] for event in event_list}

# %%
# Pick any channels you want
picks_eeg = raw.ch_names

# %%

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
    raw,
    event_id=event_id_plotting,
    picks=picks_eeg,
    plot_kwargs=plot_kwargs,
    epochs_kwargs=epochs_kwargs,
)

# Group plotting (pass a list of raws)
fig = plot_various_ERPs(
    [raw, raw_clean],
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

# Note: `plot_ERSP` is deprecated in bpn_analysis; use `plot_TFR` instead.

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
