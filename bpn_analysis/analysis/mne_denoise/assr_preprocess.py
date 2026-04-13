"""Analyze ASSR data for mne-denoise."""

# %% Imports

import gc
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import create_eog_epochs
from mne_denoise.dss import DSS, AverageBias, IterativeDSS, KurtosisDenoiser
from mne_denoise.viz import (
    plot_component_patterns,
    plot_component_score_curve,
    plot_component_summary,
    plot_component_time_series,
)

from bpn_analysis import XDFLoader
from bpn_analysis.preproc import EEGPreprocessor, compute_ica

# %% Constants & Settings

DATA_DIR = Path(
    r"\\stor1.bpn.tu-berlin.de\projects\Project_Eric\ASSR\Data\CURRENT_DATA\BIDS"
)
DERIV_DIR = DATA_DIR / "derivatives"

REMAPS = {}

EPOCH_TIMES = (-0.2, 0.8)
BASELINE = (-0.2, 0)

BANDPASS_ERP = (None, 20.0)

# Stimulus onset delay: shift trigger annotations forward by this amount so
# that t=0 in the epoch aligns with actual stimulus presentation.
# 60 ms is a typical LSL/screen-refresh latency for this setup.
TSHIFT = 0.060  # seconds

FORCE_RERUN = False  # set True to reprocess subjects even if all outputs exist

PIPELINE_NAME = "mne-denoise"

# Maps eog_cleaned dict keys → BIDS desc labels
_EOG_DESC = {
    "linear_dss": "linearDSS",
    "nonlinear_dss": "nonlinearDSS",
    "ica": "icaEOG",
}

_BIDS_ENTITY_ORDER = ("sub", "ses", "task", "acq", "run")

LOADER = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=[],  # ["x_dir", "y_dir", "z_dir"],
    target_sfreq=500.0,
)

PREPROCESSOR = EEGPreprocessor(
    LOADER,
    rng_seed=1836791205,
    asr_cutoff=20,
    exclude_labels=["muscle artifact"],
    include_labels=None,
)


# %% Helper functions


class NumpyEncoder(json.JSONEncoder):
    """Help encoding numpy arrays in JSON."""

    def default(self, obj):
        """Convert numpy arrays and floats to lists and native floats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


def clear_matplotlib_memory():
    """Clear and close all matplotlib figures and caches."""
    plt.close("all")
    matplotlib._pylab_helpers.Gcf.destroy_all()
    gc.collect()


def _parse_bids_entities(fname):
    """Return a dict of BIDS entities parsed from *fname*'s stem."""
    stem = Path(fname).name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return dict(re.findall(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", stem))


def _bids_stem(entities, desc=None, suffix="eeg"):
    """Assemble a BIDS filename stem from *entities* + optional *desc*."""
    parts = [f"{k}-{entities[k]}" for k in _BIDS_ENTITY_ORDER if k in entities]
    if desc is not None:
        parts.append(f"desc-{desc}")
    parts.append(suffix)
    return "_".join(parts)


def save_bids_derivatives(
    fname_in,
    raw_minimal,
    raw_clean,
    eog_cleaned,
    ica,
    ic_labels,
    bad_ch_dict,
    *,
    overwrite=False,
):
    """Save all ASSR preprocessing derivatives in BIDS format.

    Output layout::

        DERIV_DIR / PIPELINE_NAME / sub-XX / [ses-XX /] eeg /
            sub-XX[…]_desc-minimal_eeg.fif.gz        — bandpass only
            sub-XX[…]_desc-muscleClean_eeg.fif.gz    — after muscle ICA
            sub-XX[…]_desc-linearDSS_eeg.fif.gz      — linear DSS EOG removal
            sub-XX[…]_desc-nonlinearDSS_eeg.fif.gz   — nonlinear DSS EOG removal
            sub-XX[…]_desc-icaEOG_eeg.fif.gz         — ICA EOG removal
            sub-XX[…]_desc-muscleICA_ica.fif.gz      — fitted muscle ICA
            sub-XX[…]_iclabels.json
            sub-XX[…]_bads.json
    """
    entities = _parse_bids_entities(fname_in)

    out_dir = DERIV_DIR / PIPELINE_NAME / f"sub-{entities['sub']}"
    if "ses" in entities:
        out_dir = out_dir / f"ses-{entities['ses']}"
    out_dir = out_dir / "eeg"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fpath(desc=None, suffix="eeg", ext=".fif.gz"):
        return out_dir / (_bids_stem(entities, desc=desc, suffix=suffix) + ext)

    raw_minimal.save(fpath(desc="minimal"), overwrite=overwrite)
    raw_clean.save(fpath(desc="muscleClean"), overwrite=overwrite)
    for key, raw in eog_cleaned.items():
        raw.save(fpath(desc=_EOG_DESC[key]), overwrite=overwrite)

    ica.save(fpath(desc="muscleICA", suffix="ica"), overwrite=overwrite)

    with open(fpath(suffix="iclabels", ext=".json"), "w") as f:
        json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)
    with open(fpath(suffix="bads", ext=".json"), "w") as f:
        json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)


# %% Functions


def plot_eog_results(
    model,
    sources,
    raw,
    info,
    picks,
    data=None,
    title_prefix="DSS",
    n_components=5,
    start_idx=5000,
    end_idx=10000,
):
    """Plot DSS / IterativeDSS EOG-removal diagnostics.

    Works for both linear DSS (AverageBias) and nonlinear IterativeDSS models.

    Parameters
    ----------
    model : DSS | IterativeDSS
        Fitted denoiser.
    sources : ndarray, shape (n_components, n_times)
        Output of ``model.transform()`` on the continuous EEG data.
    raw : mne.io.Raw
        Original recording — used to access the EOG channel and sampling rate.
    info : mne.Info
        Channel info of the EEG-only data (used for topographic patterns).
    picks : ndarray of int
        Channel indices within *info* to pass to the pattern plot.
    data : Epochs | ndarray | None
        Data used to fit the model (epochs for linear, 2-D array for nonlinear).
        When provided the component time-series and summary plots are drawn.
    title_prefix : str
        Label shown in the EOG-vs-component figure title.
    n_components : int
        Number of components shown in the diagnostic plots.
    start_idx, end_idx : int
        Sample range used for the EOG-vs-component overlay plot.
    """
    # --- Component diagnostics ------------------------------------------------
    # IterativeDSS does not expose component scores; skip gracefully.
    try:
        plot_component_score_curve(model, mode="ratio", show=True)
    except ValueError:
        print("Score curve not available for this model type — skipping.")
    if data is not None:
        plot_component_time_series(
            model, data=data, n_components=n_components, show=True
        )
    plot_component_patterns(
        model, info=info, picks=picks, n_components=n_components, show=True
    )
    if data is not None:
        plot_component_summary(
            model,
            data=data,
            info=info,
            picks=picks,
            n_components=list(range(min(4, n_components))),
            show=True,
        )

    # --- EOG correlation check ------------------------------------------------
    print("Removing blink component...")
    eog_picks = mne.pick_types(raw.info, meg=False, eog=True)
    if len(eog_picks) > 0:
        eog_data = raw.get_data(picks=eog_picks[0]).flatten()
        corrs = np.array(
            [
                abs(np.corrcoef(eog_data, sources[i, :])[0, 1])
                for i in range(sources.shape[0])
            ]
        )
        for i, c in enumerate(corrs):
            print(f"Correlation (Comp {i} vs EOG): {c:.3f}")
        best_idx = int(np.argmax(corrs))
        best_corr = corrs[best_idx]
        print(f"→ Using Comp {best_idx} (highest EOG correlation: {best_corr:.3f})")
    else:
        print("No EOG channel found — falling back to Comp 0.")
        eog_data = np.zeros(sources.shape[1])
        best_idx = 0
        best_corr = 0.0

    blink_source = sources[best_idx, :]

    # --- EOG vs best-correlated component overlay ----------------------------
    # Show a time window with clear blinks, scaled and polarity-aligned.
    t_window = np.arange(start_idx, end_idx) / raw.info["sfreq"]
    eog_snippet = eog_data[start_idx:end_idx]
    comp_snippet = blink_source[start_idx:end_idx]

    flip = -1 if np.corrcoef(eog_snippet, comp_snippet)[0, 1] < 0 else 1
    scale = np.max(np.abs(eog_snippet)) / np.max(np.abs(comp_snippet))

    plt.figure(figsize=(12, 4))
    plt.plot(t_window, eog_snippet, "b", linewidth=1.5, label="EOG Channel")
    plt.plot(
        t_window,
        flip * comp_snippet * scale,
        "r",
        linewidth=1.5,
        label=f"DSS Comp {best_idx} (aligned & scaled)",
        alpha=0.8,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(
        f"{title_prefix}: Blink Peaks Aligned — Comp {best_idx} (r={best_corr:.3f})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_idx


def plot_removal_comparison(raw_original, cleaned_raws, tmin=-0.5, tmax=0.5):
    """Plot blink-locked ERP comparison across EOG-removal methods.

    Creates EOG epochs from the original and each cleaned recording, then
    shows the trial-averaged blink waveform at the most-affected frontal
    channel.  A large residual blink in a cleaned version means the method
    left artefact behind; a flat trace means good removal.

    Parameters
    ----------
    raw_original : mne.io.Raw
        Original (uncleaned) EEG-only recording.  Must contain the EOG channel.
    cleaned_raws : dict[str, mne.io.Raw]
        Mapping of method label → cleaned EEG-only raw
        (e.g. ``{"Linear DSS": raw_lin, "Nonlinear DSS": raw_nl, "ICA": raw_ica}``).
        Each raw must share the same channel layout as *raw_original*.
    tmin, tmax : float
        Epoch window around each detected blink (seconds).
    """
    colors = ["k", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    # ── Detect blinks on original data ────────────────────────────────────────
    eog_events = mne.preprocessing.find_eog_events(raw_original, ch_name="EOG")
    if len(eog_events) == 0:
        print("No blink events found — skipping comparison plot.")
        return

    # ── Build epochs from original and each cleaned raw ───────────────────────
    all_raws = {"Original": raw_original} | cleaned_raws
    epochs_dict = {}
    for label, rw in all_raws.items():
        # add EOG channel if missing (original has it; cleaned copies won't)
        if "EOG" not in rw.ch_names and "EOG" in raw_original.ch_names:
            rw = rw.copy().add_channels(
                [raw_original.copy().pick_channels(["EOG"])], force_update_info=True
            )
        ep = mne.Epochs(
            rw,
            eog_events,
            event_id=998,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        ep.pick("eeg")
        epochs_dict[label] = ep

    # ── Choose the channel with the largest blink peak in the original ────────
    orig_avg = np.abs(epochs_dict["Original"].average().data)
    best_ch_idx = int(np.argmax(orig_avg.max(axis=1)))
    ch_name = epochs_dict["Original"].ch_names[best_ch_idx]
    times = epochs_dict["Original"].times

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for (label, ep), color in zip(epochs_dict.items(), colors):
        mean = ep.average().data[best_ch_idx] * 1e6  # V → µV
        sem = ep.get_data()[:, best_ch_idx, :].std(axis=0) / np.sqrt(len(ep)) * 1e6
        ax.plot(times, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Blink-locked ERP — {ch_name}  (residual = artefact remaining)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def linear_dss_EOG_removal(raw):
    """Use present EOG channel to fit a linear DSS model for blink removal.

    Returns
    -------
    raw_clean : mne.io.Raw
        EEG-only raw with the blink DSS component removed.
    """
    # 1. Create EOG Epochs
    eog_epochs = create_eog_epochs(
        raw,
        ch_name="EOG",
        baseline=(-0.5, -0.2),
        tmin=-0.5,
        tmax=0.5,
        verbose=False,
    )
    # IMPORTANT: DSS should be fitted on the data channels (MEG) we want to clean.
    # We exclude the EOG channel itself from the model.
    eog_epochs.pick_types(eeg=True, eog=False, ecg=False)
    print(
        f"Found {len(eog_epochs)} blink events. "
        f"Using {len(eog_epochs.ch_names)} EEG channels."
    )
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, ecg=False)
    raw_picks = np.arange(len(raw_eeg.ch_names))

    # 2. Fit DSS with Trial Average Bias
    # We use AverageBias(axis='epochs') which works on pre-epoched data.
    # Note: We'll compare with CycleAverageBias later (artifact-specific approach).
    dss_eog = DSS(
        n_components=10, bias=AverageBias(axis="epochs"), return_type="sources"
    )
    dss_eog.fit(eog_epochs)

    # Transform continuous data to sources for the EOG correlation plot.
    sources = dss_eog.transform(raw_eeg)

    best_idx = plot_eog_results(
        model=dss_eog,
        sources=sources,
        raw=raw,
        info=eog_epochs.info,
        picks=raw_picks,
        data=eog_epochs,
        title_prefix="TrialAverageBias",
        n_components=10,
    )

    # Reconstruct: keep every component except the blink one.
    keep_idx = [i for i in range(sources.shape[0]) if i != best_idx]
    clean_data = dss_eog.inverse_transform(sources, component_indices=keep_idx)
    raw_clean = raw_eeg.copy()
    raw_clean._data = clean_data
    return raw_clean


def nonlinear_dss_EOG_removal(raw):
    """Remove EOG channel and fit a nonlinear DSS model for blink removal.

    Note:
    -----
    - finds the most kurtotic component blindly. Blinks usually win because they
      dominate kurtosis, but if there's a stronger non-Gaussian artifact
      (e.g., muscle bursts), that could rank higher instead. This is very
      relevant for MoBI data.

    Returns
    -------
    raw_clean : mne.io.Raw
        EEG-only raw with the blink DSS component removed.
    """
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False, ecg=False)
    raw_picks = np.arange(len(raw_eeg.ch_names))
    data = raw_eeg.get_data()

    denoiser = KurtosisDenoiser(nonlinearity="tanh")  # {'tanh', 'cube', 'gauss'}
    # 'tanh' is robust to outliers because the function saturates at large values.
    # Eye artifacts produce extreme sample values, tanh won't be destabilized by them.
    it_dss = IterativeDSS(denoiser, n_components=5, max_iter=100)
    it_dss.fit(data)
    sources = it_dss.transform(data)

    best_idx = plot_eog_results(
        model=it_dss,
        sources=sources,
        raw=raw,
        info=raw_eeg.info,
        picks=raw_picks,
        data=data,
        title_prefix="KurtosisDSS (nonlinear)",
        n_components=5,
    )

    # Reconstruct: zero out the blink component, keep the rest.
    sources_clean = sources.copy()
    sources_clean[best_idx, :] = 0
    clean_data = it_dss.inverse_transform(sources_clean)
    raw_clean = raw_eeg.copy()
    raw_clean._data = clean_data
    return raw_clean


def ICA_EOG_removal(
    raw,
    filter_bands_ica=(1.0, 100.0),
    notch_freqs=(50, 100, 150),
    downsample_ica=250,
    thresh=0.7,
    rng_seed=None,
):
    """Remove eye components from *raw* by fitting a dedicated ICA.

    Fits a fresh ICA on *raw* and excludes components labelled as eye
    artifacts by ICLabel.  Kept separate from the preprocessing ICA so
    that the decomposition is optimised for EOG removal rather than
    muscle rejection, ensuring a fair comparison with the DSS methods.

    Returns
    -------
    raw_clean : mne.io.Raw
        EEG-only raw with eye components removed.
    """
    ica, _ = compute_ica(
        raw,
        filter_bands_ica=filter_bands_ica,
        notch_freqs=notch_freqs,
        downsample_ica=downsample_ica,
        thresh=thresh,
        rng_seed=rng_seed,
        exclude_labels=["eye"],
    )
    return ica.apply(raw.copy().pick("eeg"))


def run_analysis(raw):
    """Run the EOG-removal comparison pipeline on an already-loaded *raw*.

    Pipeline:
      1. :data:`PREPROCESSOR` — bad channels → ASR → muscle-artifact ICA
         → ``raw_clean`` (common baseline for all EOG methods)
      2. Three EOG-removal methods applied in parallel to ``raw_clean``:
         - Linear DSS  (trial-average bias)
         - Nonlinear DSS  (kurtosis-based)
         - ICA  (fresh fit, eye-label exclusion)

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Pre-loaded recording with EOG channel type already set.

    Returns
    -------
    raw_minimal : mne.io.Raw
        Bandpass-filtered recording before any artifact removal.
    raw_clean : mne.io.Raw
        Muscle-ICA cleaned recording — common input to all EOG methods.
    eog_cleaned : dict[str, mne.io.Raw]
        ``{"linear_dss": ..., "nonlinear_dss": ..., "ica": ...}``
    ica : mne.preprocessing.ICA
    ic_labels : dict
    bad_ch_dict : dict
    """
    # Stage 1: shared preprocessing (muscle artifacts removed)
    raw_minimal, raw_clean, _, ica, ic_labels, bad_ch_dict = PREPROCESSOR.run_raw(raw)

    # Stage 2: EOG removal — each method receives the same raw_clean
    eog_cleaned = {
        "linear_dss": linear_dss_EOG_removal(raw_clean),
        "nonlinear_dss": nonlinear_dss_EOG_removal(raw_clean),
        "ica": ICA_EOG_removal(raw_clean, rng_seed=PREPROCESSOR.rng_seed),
    }

    return raw_minimal, raw_clean, eog_cleaned, ica, ic_labels, bad_ch_dict


def main():
    """Run main function."""
    for fname_in in DATA_DIR.rglob("*.xdf"):
        raw = LOADER.load(fname_in)
        if "EOG" not in raw.ch_names:
            print(f"[SKIP] {fname_in.name} — no EOG channel")
            continue
        raw.set_channel_types({"EOG": "eog"})

        raw_minimal, raw_clean, eog_cleaned, ica, ic_labels, bad_ch_dict = run_analysis(
            raw
        )

        save_bids_derivatives(
            fname_in,
            raw_minimal,
            raw_clean,
            eog_cleaned,
            ica,
            ic_labels,
            bad_ch_dict,
            overwrite=FORCE_RERUN,
        )

        plot_removal_comparison(raw_minimal, eog_cleaned)
        clear_matplotlib_memory()


# %% Main
if __name__ == "__main__":
    # copy_data()
    main()

# %%
