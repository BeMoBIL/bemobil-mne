"""Show how to use the `preproc` module."""

# %%
# Imports

import json
import time
from pathlib import Path

import mne
from mne.datasets import eegbci

import bemobil_mne
from bemobil_mne.preproc import EEGPreprocessor
from bemobil_mne.preproc.utils import get_descriptor

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
    Path(bemobil_mne.__file__).parent.parent / "reports" / "preprocess_plot_example"
)
save_path.mkdir(exist_ok=True, parents=True)
fname_out = save_path / "testfile"
seed = 869330571  # good practice: pick a random seed via secrets.randbits(31)

# %%
# Run preprocessing

# This will create files in the save_path directory; see the docstring of
# `EEGPreprocessor` for details (e.g., `print(help(EEGPreprocessor))`).
#
# `run_raw` returns a 10-tuple:
#   (raw_minimal, raw_clean, raw_asr, raw_subset, ica, ic_labels,
#    dipoles, residuals, trans, bad_ch_dict)
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
# or with custom ASR settings via asr={"cutoff": 10}, and/or without ICA
# via fit_ica=False. Set `fname_out=None` to skip saving.

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
    import matplotlib.pyplot as plt

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
