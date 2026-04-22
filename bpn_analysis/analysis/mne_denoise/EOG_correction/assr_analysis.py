"""Analyze ASSR data for mne-denoise."""

# %% Imports

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import create_eog_epochs, find_eog_events
from mne_denoise.dss import DSS, AverageBias, IterativeDSS, KurtosisDenoiser
from mne_denoise.viz import (
    plot_component_patterns,
    plot_component_score_curve,
    plot_component_summary,
    plot_component_time_series,
)
from mne_denoise.viz.components import _get_scores

from bpn_analysis.analysis.mne_denoise.EOG_correction.assr_cfg import (
    _BIDS_ENTITY_ORDER,
    _EOG_DESC,
    DATA_DIR,
    DERIV_DIR,
    FORCE_RERUN,
    LOADER,
    MODE,
    PIPELINE_NAME,
    PREPROCESSOR,
    README_DESTINATION,
)
from bpn_analysis.io.utils import NumpyEncoder
from bpn_analysis.preproc import compute_ica
from bpn_analysis.viz.utils import clear_matplotlib_memory

# %% Helper functions


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
    entities,
    raw_minimal,
    raw_clean,
    eog_cleaned,
    ica,
    ic_labels,
    bad_ch_dict,
    blink_df,
    peak_info,
    *,
    overwrite=False,
):
    """Save all ASSR preprocessing derivatives in BIDS format.

    Output layout::

        DERIV_DIR / PIPELINE_NAME / sub-XX / [ses-XX /] eeg /
            sub-XX[…]_desc-minimal_eeg.fif.gz           : bandpass only
            sub-XX[…]_desc-preproc-clean_eeg.fif.gz     : after muscle ICA
            sub-XX[…]_desc-linearDSS_eeg.fif.gz         : linear DSS EOG removal
            sub-XX[…]_desc-nonlinearDSS_eeg.fif.gz      : nonlinear DSS EOG removal
            sub-XX[…]_desc-icaEOG_eeg.fif.gz            : ICA EOG removal
            sub-XX[…]_desc-muscleICA_ica.fif.gz         : fitted muscle ICA
            sub-XX[…]_iclabels.json
            sub-XX[…]_bads.json
            sub-XX[…]_desc-blinkAmplitudes_channels.tsv : per-blink amplitude table
            sub-XX[…]_desc-blinkAmplitudes_channels.json: peak channel / latency
    """
    out_dir = DERIV_DIR / PIPELINE_NAME / f"sub-{entities['sub']}"
    if "ses" in entities:
        out_dir = out_dir / f"ses-{entities['ses']}"
    out_dir = out_dir / "eeg"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fpath(desc=None, suffix="eeg", ext=".fif.gz"):
        return out_dir / (_bids_stem(entities, desc=desc, suffix=suffix) + ext)

    raw_minimal.save(fpath(desc="minimal"), overwrite=overwrite)
    raw_clean.save(fpath(desc="preproc-clean"), overwrite=overwrite)
    for key, raw in eog_cleaned.items():
        raw.save(fpath(desc=_EOG_DESC[key]), overwrite=overwrite)

    ica.save(fpath(desc="muscleICA", suffix="ica"), overwrite=overwrite)

    with open(fpath(suffix="iclabels", ext=".json"), "w") as f:
        json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)
    with open(fpath(suffix="bads", ext=".json"), "w") as f:
        json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)

    tsv_path = fpath(desc="blinkAmplitudes", suffix="channels", ext=".tsv")
    if overwrite or not tsv_path.exists():
        blink_df.to_csv(tsv_path, sep="\t", index=False)
    with open(fpath(desc="blinkAmplitudes", suffix="channels", ext=".json"), "w") as f:
        json.dump(peak_info, f, indent=4)


def compute_blink_amplitudes(raw_original, cleaned_raws, tmin=-0.5, tmax=0.5):
    """Per-blink amplitude at the peak blink latency for each recording.

    Determines the EEG channel and time point that show the largest average
    blink artifact in *raw_original*, then extracts the single-trial amplitude
    at that (channel, latency) coordinate from every recording.

    Parameters
    ----------
    raw_original : mne.io.Raw
        Original (uncleaned) recording.  Must carry blink annotations
        (label ``"blink"``) added before this call.
    cleaned_raws : dict[str, mne.io.Raw]
        Mapping of method label → cleaned raw
        (e.g. ``{"linear_dss": raw_lin, "nonlinear_dss": raw_nl, "ica": raw_ica}``).
    tmin, tmax : float
        Epoch window around each blink (seconds).

    Returns
    -------
    df : pd.DataFrame
        Shape ``(n_blinks, 1 + len(cleaned_raws))``.
        Columns: ``"original"`` followed by each key in *cleaned_raws*.
        Each cell is the amplitude in µV at the peak-blink (channel, latency)
        for that blink event.  A good cleaning method produces values near zero.
    peak_info : dict
        ``{"channel": str, "latency_s": float}``: the coordinate used.
    """
    # blink events from annotations
    events, ids = mne.events_from_annotations(raw_original, verbose=False)
    blink_id = {"blink": ids.get("blink", 998)}
    eog_events, event_id_map = mne.events_from_annotations(
        raw_original, event_id=blink_id, verbose=False
    )

    if len(eog_events) == 0:
        raise ValueError("No blink annotations found on raw_original.")

    # build epochs for every recording
    all_raws = {"original": raw_original} | cleaned_raws
    epochs_dict = {}
    for label, rw in all_raws.items():
        ep = mne.Epochs(
            rw,
            eog_events,
            event_id=event_id_map,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        ep.pick("eeg")
        epochs_dict[label] = ep

    # peak coordinate from the original's average ERP
    orig_avg = epochs_dict["original"].average().data  # (n_ch, n_times)
    best_ch_idx = int(np.argmax(np.abs(orig_avg).max(axis=1)))
    best_t_idx = int(np.argmax(np.abs(orig_avg[best_ch_idx])))
    ch_name = epochs_dict["original"].ch_names[best_ch_idx]
    peak_latency_s = float(epochs_dict["original"].times[best_t_idx])

    # extract single-trial amplitude at that coordinate
    records = {}
    for label, ep in epochs_dict.items():
        # ep.get_data() → (n_epochs, n_ch, n_times)
        records[label] = ep.get_data()[:, best_ch_idx, best_t_idx] * 1e6  # V → µV

    df = pd.DataFrame(records)
    peak_info = {"channel": ch_name, "latency_s": peak_latency_s}
    return df, peak_info


# %% EOG correction


def linear_dss_EOG_removal(raw, out_dir, entities=None):
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
        out_dir=out_dir,
        entities=entities,
        desc="linearDSS",
    )

    # Reconstruct: keep every component except the blink one.
    keep_idx = [i for i in range(sources.shape[0]) if i != best_idx]
    clean_data = dss_eog.inverse_transform(sources, component_indices=keep_idx)
    raw_clean = raw_eeg.copy()
    raw_clean._data = clean_data
    return raw_clean


def nonlinear_dss_EOG_removal(raw, out_dir, entities=None):
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
        out_dir=out_dir,
        entities=entities,
        desc="nonlinearDSS",
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
    ica, ic_labels = compute_ica(
        raw.copy().pick("eeg"),
        filter_bands_ica=filter_bands_ica,
        notch_freqs=notch_freqs,
        downsample_ica=downsample_ica,
        thresh=thresh,
        rng_seed=rng_seed,
        exclude_labels=["eye blink"],
    )
    for idx, (lbl, prob) in enumerate(
        zip(ic_labels["labels"], ic_labels["y_pred_proba"])
    ):
        marker = " ← excluded" if idx in ica.exclude else ""
        print(f"  IC{idx:03d}: {lbl:<20s} prob={prob:.3f}{marker}")
    print(f"ICA EOG: excluding {len(ica.exclude)} component(s): {ica.exclude}")

    raw_ica = ica.apply(raw.copy().pick("eeg"))

    return raw_ica


# %% Plotting


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
    out_dir=None,
    entities=None,
    desc="dss",
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
        Original recording: used to access the EOG channel and sampling rate.
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
    out_dir : path-like | None
        Directory where all figures are saved.  If None figures are shown
        interactively (not recommended for batch runs).
    """

    def _save_or_show(fig, plot_name):
        if out_dir is not None:
            if entities is not None:
                fname = (
                    _bids_stem(entities, desc=f"{desc}-{plot_name}", suffix="fig")
                    + ".png"
                )
            else:
                fname = f"{desc}_{plot_name}.png"
            fig.savefig(Path(out_dir) / fname, dpi=150)
            plt.close(fig)

    # --- Component diagnostics ------------------------------------------------
    # IterativeDSS does not expose component scores; skip gracefully.
    try:
        fig = plot_component_score_curve(model, mode="ratio", show=False)
        _save_or_show(fig, "scoreCurve")
    except ValueError:
        print("Score curve not available for this model type: skipping.")
    if data is not None:
        fig = plot_component_time_series(
            model, data=data, n_components=n_components, show=False
        )
        _save_or_show(fig, "timeSeries")
    fig = plot_component_patterns(
        model, info=info, picks=picks, n_components=n_components, show=False
    )
    _save_or_show(fig, "patterns")
    if data is not None:
        fig = plot_component_summary(
            model,
            data=data,
            info=info,
            picks=picks,
            n_components=list(range(min(4, n_components))),
            show=False,
        )
        _save_or_show(fig, "summary")

    try:
        scores = _get_scores(model)
        mean_score = np.mean(scores)
        candidates = [i for i, s in enumerate(scores) if s >= mean_score]
    except Exception:
        print(
            "Component scores not available for this model type:"
            " skipping score-based candidate selection."
        )
        candidates = list(range(sources.shape[0]))

    # --- EOG correlation check ------------------------------------------------
    print("Removing blink component...")
    eog_picks = mne.pick_types(raw.info, meg=False, eog=True)
    if len(eog_picks) > 0:
        eog_data = raw.get_data(picks=eog_picks[0]).flatten()
        corrs = np.array(
            [abs(np.corrcoef(eog_data, sources[i, :])[0, 1]) for i in candidates]
        )
        for i, c in enumerate(corrs):
            print(f"Correlation (Comp {i} vs EOG): {c:.3f}")
        best_idx = int(np.argmax(corrs))
        best_corr = corrs[best_idx]
        print(f"→ Using Comp {best_idx} (highest EOG correlation: {best_corr:.3f})")
    else:
        print("No EOG channel found: falling back to Comp 0.")
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

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_window, eog_snippet, "b", linewidth=1.5, label="EOG Channel")
    ax.plot(
        t_window,
        flip * comp_snippet * scale,
        "r",
        linewidth=1.5,
        label=f"DSS Comp {best_idx} (aligned & scaled)",
        alpha=0.8,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(
        f"{title_prefix}: Blink Peaks Aligned: Comp {best_idx} (r={best_corr:.3f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "blinkOverlay")

    return best_idx


def plot_blink_erp_comparison(data, out_path):
    """Blink-locked ERP comparison across EOG-removal methods.

    Works for both single-subject and group data.

    Parameters
    ----------
    data : dict[str, mne.Epochs | list[mne.Evoked]]
        Per-method data keyed by label. Values are either:
        - ``mne.Epochs``: single subject: SEM computed across trials
        - ``list[mne.Evoked]``: group: SEM computed across subjects
    out_path : path-like, optional
        Save destination. If given the figure is saved and closed; otherwise shown.
    """
    colors = ["k", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    # normalize: detect single-subject (Epochs) vs group (list[Evoked])
    is_group = isinstance(next(iter(data.values())), list)

    stats = {}  # label → (mean_uv, sem_uv, times, ch_names)
    for label, val in data.items():
        if is_group:
            grand = mne.grand_average(val)
            ch_names = grand.ch_names
            times = grand.times
            mean_uv = grand.data * 1e6
            sub_mat = (
                np.stack([e.copy().pick_channels(list(ch_names)).data for e in val])
                * 1e6
            )  # (n_sub, n_ch, n_times)
            sem_uv = sub_mat.std(axis=0) / np.sqrt(len(val))
        else:
            ch_names = val.ch_names
            times = val.times
            mean_uv = val.average().data * 1e6
            sem_uv = val.get_data().std(axis=0) / np.sqrt(len(val)) * 1e6
        stats[label] = (mean_uv, sem_uv, times, ch_names)

    # auto-select EEG channel with largest blink peak in "original"
    orig_label = "original" if "original" in stats else next(iter(stats))
    orig_mean, _, times, orig_chs = stats[orig_label]
    best_ch_idx = int(np.argmax(np.abs(orig_mean).max(axis=1)))
    ch_name = orig_chs[best_ch_idx]

    _, ax = plt.subplots(figsize=(10, 4))
    for (label, (mean_uv, sem_uv, _, ch_names)), color in zip(stats.items(), colors):
        idx = list(ch_names).index(ch_name)
        ax.plot(times, mean_uv[idx], color=color, linewidth=1.8, label=label)
        ax.fill_between(
            times,
            mean_uv[idx] - sem_uv[idx],
            mean_uv[idx] + sem_uv[idx],
            color=color,
            alpha=0.15,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    prefix = "Group " if is_group else ""
    ax.set_title(
        f"{prefix}Blink-locked ERP: {ch_name}  (residual = artefact remaining)"
    )
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_blink_amplitude_comparison(df, out_path):
    """Violin plot of per-blink amplitudes across EOG-removal methods.

    Works for both single-subject and group data.  If *df* has a ``"sub"``
    column each unique value gets its own panel; otherwise a single panel is
    drawn.

    Parameters
    ----------
    df : pd.DataFrame
        Per-blink amplitudes in µV.  Method columns hold scalar values.
        An optional ``"sub"`` column panels the plot by subject.
    out_path : path-like, optional
        Save destination.  If given the figure is saved and closed; otherwise shown.
    """
    # normalize: single-subject df has no 'sub' column
    is_group = "sub" in df.columns
    if not is_group:
        df = df.copy()
        df.insert(0, "sub", "")

    method_cols = [c for c in df.columns if c != "sub"]
    groups = list(df.groupby("sub", sort=False))
    n_panels = len(groups)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(4 * n_panels, 5), sharey=True, squeeze=False
    )
    for ax, (sub, sub_df) in zip(axes[0], groups):
        vals = [sub_df[col].dropna().values for col in method_cols]
        parts = ax.violinplot(vals, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        for i, col in enumerate(method_cols, start=1):
            ax.scatter(i, sub_df[col].mean(), color="k", zorder=3, s=30)
        ax.set_xticks(range(1, len(method_cols) + 1))
        ax.set_xticklabels(method_cols, rotation=25, ha="right", fontsize=8)
        if sub:
            ax.set_title(f"sub-{sub}", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0][0].set_ylabel("amplitude at blink peak (µV)")
    prefix = "group " if is_group else ""
    fig.suptitle(f"{prefix}blink amplitude: residual per method", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def _expected_outputs(fname_in):
    """Return all output paths that run_single_subject would produce."""
    entities = _parse_bids_entities(fname_in)
    out_dir = DERIV_DIR / PIPELINE_NAME / f"sub-{entities['sub']}"
    if "ses" in entities:
        out_dir = out_dir / f"ses-{entities['ses']}"
    out_dir = out_dir / "eeg"

    def fpath(desc=None, suffix="eeg", ext=".fif.gz"):
        return out_dir / (_bids_stem(entities, desc=desc, suffix=suffix) + ext)

    return [
        fpath(desc="minimal"),
        fpath(desc="preproc-clean"),
        *(fpath(desc=desc) for desc in _EOG_DESC.values()),
        fpath(desc="muscleICA", suffix="ica"),
        fpath(desc="blinkERP-comparison", suffix="fig", ext=".png"),
        fpath(desc="blinkAmplitude-comparison", suffix="fig", ext=".png"),
    ]


def run_single_subject(fname_in):
    """Run the EOG-removal comparison pipeline on an already-loaded *raw*.

    Pipeline:
      1. :data:`PREPROCESSOR`: bad channels → ASR → muscle-artifact ICA
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
        Muscle-ICA cleaned recording: common input to all EOG methods.
    eog_cleaned : dict[str, mne.io.Raw]
        ``{"linear_dss": ..., "nonlinear_dss": ..., "ica": ...}``
    ica : mne.preprocessing.ICA
    ic_labels : dict
    bad_ch_dict : dict
    """
    if not FORCE_RERUN and all(p.exists() for p in _expected_outputs(fname_in)):
        print(f"[SKIP] {fname_in.name}: all outputs present")
        return

    print(f"\n=== Processing {fname_in} ===")
    raw = LOADER.load(fname_in)
    if "EOG" not in raw.ch_names:
        print(f"[SKIP] {fname_in.name}: no EOG channel")
        return

    # resolve output directory early so plots land in the right place
    entities = _parse_bids_entities(fname_in)
    sub_dir = DERIV_DIR / PIPELINE_NAME / f"sub-{entities['sub']}"
    if "ses" in entities:
        sub_dir = sub_dir / f"ses-{entities['ses']}"
    sub_dir = sub_dir / "eeg"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # get EOG channel
    raw.set_channel_types({"EOG": "eog"})
    raw_eog = raw.copy()
    raw_eog.pick_types(eog=True)
    raw.pick_types(eeg=True, eog=False, ecg=False)

    # find all blink events
    blink_events = find_eog_events(raw_eog, ch_name="EOG", verbose=False)
    blink_annots = mne.annotations_from_events(
        blink_events, sfreq=raw_eog.info["sfreq"], event_desc={998: "blink"}
    )
    print(f"Found {len(blink_annots)} blink events")

    # shared preprocessing
    raw_minimal, raw_clean, _, ica, ic_labels, bad_ch_dict = PREPROCESSOR.run_raw(raw)

    bad = [d.upper().startswith("BAD") for d in raw_minimal.annotations.description]
    bad_annots = mne.Annotations(
        raw_minimal.annotations.onset[bad],
        raw_minimal.annotations.duration[bad],
        raw_minimal.annotations.description[bad],
        orig_time=raw_minimal.annotations.orig_time,
    )

    # reintroduce the EOG channel; keep BAD annotations from preprocessing so
    # as well as blinks. Other annots can be dropped.
    for raw_temp in (raw, raw_minimal, raw_clean):
        raw_temp.set_annotations(bad_annots + blink_annots)
        raw_temp.add_channels([raw_eog.copy()], force_update_info=True)

    # EOG artifact removal
    eog_cleaned = {
        "linear_dss": linear_dss_EOG_removal(
            raw_clean, out_dir=sub_dir, entities=entities
        ),
        "nonlinear_dss": nonlinear_dss_EOG_removal(
            raw_clean, out_dir=sub_dir, entities=entities
        ),
        "ica": ICA_EOG_removal(raw_clean, rng_seed=PREPROCESSOR.rng_seed),
    }

    # get peak amplitude for every blink
    df, peak_info = compute_blink_amplitudes(raw_minimal, eog_cleaned)

    # save derivatives
    save_bids_derivatives(
        entities,
        raw_minimal,
        raw_clean,
        eog_cleaned,
        ica,
        ic_labels,
        bad_ch_dict,
        df,
        peak_info,
        overwrite=FORCE_RERUN,
    )

    # compute blink epochs for ERP plot
    _, ids = mne.events_from_annotations(raw_minimal, verbose=False)
    blink_ev, ev_id = mne.events_from_annotations(
        raw_minimal, event_id={"blink": ids.get("blink", 998)}, verbose=False
    )
    erp_data = {}
    for label, rw in ({"original": raw_minimal} | eog_cleaned).items():
        ep = mne.Epochs(
            rw,
            blink_ev,
            event_id=ev_id,
            tmin=-0.5,
            tmax=0.5,
            baseline=None,
            preload=True,
            verbose=False,
        )
        ep.pick("eeg")
        erp_data[label] = ep

    plot_blink_erp_comparison(
        erp_data,
        out_path=sub_dir
        / f"{_bids_stem(entities, desc='blinkERP-comparison', suffix='fig')}.png",
    )
    plot_blink_amplitude_comparison(
        df,
        out_path=sub_dir
        / f"{_bids_stem(entities, desc='blinkAmplitude-comparison', suffix='fig')}.png",
    )
    clear_matplotlib_memory()


def run_group():
    """Load per-subject blink amplitude TSVs and produce a group-level report.

    Reads every ``*_desc-blinkAmplitudes_channels.tsv`` saved by the single-
    subject pipeline, stacks them into one long DataFrame, computes per-subject
    summary statistics, saves both to the derivatives root, and plots a violin
    comparison across EOG-removal methods.

    Outputs written to ``DERIV_DIR / PIPELINE_NAME /``::

        group_desc-blinkAmplitudes_channels.tsv    : all trials, all subjects
        group_desc-blinkAmplitudesSummary_channels.tsv: per-subject means / RMS
    """
    tsv_files = sorted(
        (DERIV_DIR / PIPELINE_NAME).rglob("*_desc-blinkAmplitudes_channels.tsv")
    )
    if not tsv_files:
        print("No per-subject blink amplitude files found: run single-subject first.")
        return

    # load and stack
    dfs = []
    for f in tsv_files:
        entities = _parse_bids_entities(f)
        sub_df = pd.read_csv(f, sep="\t")
        sub_df.insert(0, "sub", entities.get("sub", "unknown"))
        dfs.append(sub_df)
    group_df = pd.concat(dfs, ignore_index=True)

    method_cols = [c for c in group_df.columns if c != "sub"]

    # per-subject summary: mean amplitude and RMS per method
    summary_rows = []
    for sub, sub_df in group_df.groupby("sub"):
        row = {"sub": sub}
        for col in method_cols:
            vals = sub_df[col].dropna()
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_rms"] = float(np.sqrt((vals**2).mean()))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # save
    out_dir = DERIV_DIR / PIPELINE_NAME
    group_df.to_csv(
        out_dir / "group_desc-blinkAmplitudes_channels.tsv", sep="\t", index=False
    )
    summary_df.to_csv(
        out_dir / "group_desc-blinkAmplitudesSummary_channels.tsv",
        sep="\t",
        index=False,
    )

    plot_blink_amplitude_comparison(
        group_df, out_path=out_dir / "group_blink-amplitude-comparison.png"
    )

    # group ERP plot: load per-subject FIFs and compute evokeds
    minimal_fifs = sorted(
        (DERIV_DIR / PIPELINE_NAME).rglob("*_desc-minimal_eeg.fif.gz")
    )
    if minimal_fifs:
        evokeds = {label: [] for label in ["original", *_EOG_DESC]}
        for minimal_fif in minimal_fifs:
            raw_min = mne.io.read_raw_fif(minimal_fif, preload=True, verbose=False)
            _, ann_ids = mne.events_from_annotations(raw_min, verbose=False)
            if "blink" not in ann_ids:
                continue
            blink_ev, ev_id = mne.events_from_annotations(
                raw_min, event_id={"blink": ann_ids["blink"]}, verbose=False
            )
            if len(blink_ev) == 0:
                continue

            sub_fifs = {"original": minimal_fif} | {
                key: next(
                    iter(minimal_fif.parent.glob(f"*_desc-{desc}_eeg.fif.gz")), None
                )
                for key, desc in _EOG_DESC.items()
            }
            for label, fif_path in sub_fifs.items():
                if fif_path is None:
                    continue
                rw = (
                    raw_min
                    if label == "original"
                    else mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
                )
                ep = mne.Epochs(
                    rw,
                    blink_ev,
                    event_id=ev_id,
                    tmin=-0.5,
                    tmax=0.5,
                    baseline=None,
                    preload=True,
                    verbose=False,
                )
                ep.pick("eeg")
                if len(ep) > 0:
                    evokeds[label].append(ep.average())

        evokeds = {k: v for k, v in evokeds.items() if v}
        if evokeds:
            plot_blink_erp_comparison(
                evokeds, out_path=out_dir / "group_blinkERP-comparison.png"
            )


# %% Main driver


def main():
    """Run main function."""
    if MODE in ("single", True):
        for fname_in in DATA_DIR.rglob("*.xdf"):
            if "walk" not in fname_in.name:
                print(f"[SKIP] {fname_in.name}: not a walk session")
                continue
            run_single_subject(fname_in)

    if MODE in ("group", True):
        run_group()

    # save README in repository to the destination
    if README_DESTINATION:
        readme_src = Path(__file__).parent / "README.md"
        readme_dst = README_DESTINATION
        readme_dst.parent.mkdir(parents=True, exist_ok=True)
        readme_dst.write_text(readme_src.read_text())
        print(f"Saved README to {readme_dst}")


# %% Main
if __name__ == "__main__":
    main()

# %%
