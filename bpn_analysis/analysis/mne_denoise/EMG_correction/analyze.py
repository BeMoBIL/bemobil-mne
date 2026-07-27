"""
Spot-rotation EMG correction.

compare linear DSS, nonlinear DSS, ICA-head, ICA+neck.
- DSS methods: linear (AverageBias) vs nonlinear (KurtosisDSS)
- ICA methods: head-only vs head+neck channels
"""

# %% Imports

import json
import re
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from mne_denoise.dss import DSS, AverageBias, IterativeDSS, KurtosisDenoiser
from mne_denoise.viz import (
    plot_component_patterns,
    plot_component_score_curve,
    plot_component_summary,
    plot_component_time_series,
)
from mne_denoise.viz.components import _get_scores
from scipy.signal import butter, filtfilt, hilbert, welch

from bpn_analysis.analysis.mne_denoise.EMG_correction.cfg import (
    _BIDS_ENTITY_ORDER,
    _EMG_DESC,
    ALPHA_BETA_BAND,
    BIDS_ROOT,
    CHANNELS_ABSENT,
    DERIV_DIR,
    EMG_BAND,
    FORCE_RERUN,
    MODE,
    N_NECK_PCA_COMPONENTS,
    NECK_CHANNEL_EXCLUDE,
    NECK_CHANNEL_PREFIX,
    NECK_CORR_THRESHOLD,
    PIPELINE_NAME,
    PREPROCESSOR,
    README_DESTINATION,
    RESPONDER_THRESHOLD_PCT,
    RV_THRESHOLD,
    SESSION,
    SKIP_THESE,
    TARGET_SFREQ,
    TASK,
)
from bpn_analysis.io.utils import NumpyEncoder
from bpn_analysis.preproc import compute_dipolarity, compute_ica, compute_mi_reduction
from bpn_analysis.viz.utils import clear_matplotlib_memory

# %% Helper functions


def _parse_bids_entities(fname) -> dict:
    """Parse BIDS entities from a filename stem (used for derivative files)."""
    stem = Path(fname).name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return dict(re.findall(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", stem))


def _entities_from_bids_path(bids_path: BIDSPath) -> dict:
    """Return {sub, ses, task, ...} from a BIDSPath (only populated entities)."""
    mapping = {
        "sub": bids_path.subject,
        "ses": bids_path.session,
        "task": bids_path.task,
        "acq": bids_path.acquisition,
        "run": bids_path.run,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _detect_neck_channels(raw: mne.io.BaseRaw) -> list[str]:
    """Return neck channel names, excluding reference/absent electrodes."""
    return [
        ch
        for ch in raw.ch_names
        if ch.startswith(NECK_CHANNEL_PREFIX) and ch not in NECK_CHANNEL_EXCLUDE
    ]


def _to_bv_name(eloc_name: str) -> str:
    """Map electrodes.tsv name to raw channel name.

    Example
    -------
    (e.g. 'g1') → 'BrainVision RDA_G01'
    """
    prefix = eloc_name[0].upper()
    num = int(eloc_name[1:])
    return f"BrainVision RDA_{prefix}{num:02d}"


def _load_bids_raw(bids_path: BIDSPath) -> mne.io.BaseRaw:
    """Read a BIDS recording, set montage from electrodes.tsv, resample to TARGET_SFREQ.

    mne-bids leaves all positions as NaN because the electrodes.tsv uses short
    names (g1, n1, …) that don't match the raw channel names (BrainVision RDA_G01, …).
    We load the file manually, remap names, and apply the 90° coordinate rotation
    confirmed for this dataset: x_mne = -y_eloc, y_mne = x_eloc, z_mne = z_eloc.
    """
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    absent = [ch for ch in CHANNELS_ABSENT if ch in raw.ch_names]
    if absent:
        raw.drop_channels(absent)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    elec_path = BIDSPath(
        subject=bids_path.subject,
        session=bids_path.session,
        suffix="electrodes",
        extension=".tsv",
        root=bids_path.root,
    ).fpath
    if elec_path.exists():
        elec = pd.read_csv(elec_path, sep="\t")
        ch_pos = {}
        for _, row in elec.iterrows():
            try:
                coords = np.array([-row["y"], row["x"], row["z"]], dtype=float) / 1000
                if not np.isnan(coords).any():
                    ch_pos[_to_bv_name(row["name"])] = coords
            except (ValueError, TypeError, KeyError):
                continue

        # write positions directly into info['chs'] — set_montage without fiducials
        # does not reliably update ch['loc'], which is what mne-icalabel reads
        n_set = 0
        for ch in raw.info["chs"]:
            if ch["ch_name"] in ch_pos:
                ch["loc"][:3] = ch_pos[ch["ch_name"]]
                n_set += 1
        print(
            f"  Set positions for {n_set}/{len(raw.ch_names)} "
            "channels from electrodes.tsv"
        )
    else:
        print(
            "  [WARN] electrodes.tsv not found for"
            f" {bids_path.subject}/{bids_path.session}"
        )

    return raw


def _bids_stem(entities, desc=None, suffix="eeg"):
    parts = [f"{k}-{entities[k]}" for k in _BIDS_ENTITY_ORDER if k in entities]
    if desc is not None:
        parts.append(f"desc-{desc}")
    parts.append(suffix)
    return "_".join(parts)


def _save_fig(fig, out_dir, entities, desc, suffix="fig"):
    fname = _bids_stem(entities, desc=desc, suffix=suffix) + ".png"
    fig.savefig(Path(out_dir) / fname, dpi=150)
    plt.close(fig)


# %% Neck channel processing


def compute_neck_pca_reference(neck_data, n_components=N_NECK_PCA_COMPONENTS):
    """Extract the most EMG-like component from neck channels via PCA + kurtosis.

    PCA captures the dominant shared variance across the neck array (common EMG).
    Kurtosis selects the most impulsive component, which is the hallmark of EMG
    bursts, even if it is not PC1 by explained variance.

    Parameters
    ----------
    neck_data : ndarray, shape (n_neck_ch, n_times)
        Raw or lightly filtered neck-channel data.
    n_components : int
        Number of PCs to evaluate.

    Returns
    -------
    neck_ref : ndarray, shape (n_times,)
        Unit-variance reference signal (PC with highest excess kurtosis).
    pca_info : dict
        ``explained_variance_ratio``, ``kurtosis_values``, ``selected_component``.
    """
    data = neck_data - neck_data.mean(axis=1, keepdims=True)
    n_comp = min(n_components, data.shape[0])

    # economy SVD: U (n_ch, n_comp), S (n_comp,), Vt (n_comp, n_times)
    _, S, Vt = np.linalg.svd(data, full_matrices=False)
    S, Vt = S[:n_comp], Vt[:n_comp]
    scores = S[:, np.newaxis] * Vt  # (n_comp, n_times) — PC score time series

    total_var = np.sum(S**2)
    exp_var_ratio = (S**2 / total_var).tolist()

    def _excess_kurtosis(x):
        x = x - x.mean()
        m2 = np.mean(x**2)
        m4 = np.mean(x**4)
        return float(m4 / (m2**2 + 1e-12) - 3.0)

    kurtosis_values = [_excess_kurtosis(scores[i]) for i in range(n_comp)]
    best_idx = int(np.argmax(kurtosis_values))
    neck_ref = scores[best_idx]
    neck_ref = neck_ref / (np.std(neck_ref) + 1e-12)

    print(
        f"Neck PCA: selected PC{best_idx} (kurtosis={kurtosis_values[best_idx]:.2f}, "
        f"explained var={exp_var_ratio[best_idx]:.3f})"
    )
    return neck_ref, {
        "explained_variance_ratio": exp_var_ratio,
        "kurtosis_values": kurtosis_values,
        "selected_component": best_idx,
    }


def _make_burst_events(neck_ref, sfreq, tmin=-0.3, tmax=0.3, threshold_sd=2.0):
    """MNE event array of EMG burst onsets detected from neck reference envelope.

    Parameters
    ----------
    neck_ref : ndarray, shape (n_times,)
        Neck PCA reference (unit variance).
    sfreq : float
    tmin, tmax : float
        Epoch window (used only to exclude onsets too close to data edges).
    threshold_sd : float
        Envelope threshold in multiples of its own standard deviation above mean.

    Returns
    -------
    events : ndarray, shape (n_bursts, 3)
        MNE-format events (sample, 0, 1).  Empty if no bursts found.
    """
    # bandpass to EMG band before envelope so burst detection is specific to
    # muscle activity
    b, a = butter(
        4, [EMG_BAND[0] / (sfreq / 2), EMG_BAND[1] / (sfreq / 2)], btype="bandpass"
    )
    neck_emg = filtfilt(b, a, neck_ref)
    envelope = np.abs(hilbert(neck_emg))
    k = max(1, int(0.05 * sfreq))  # 50 ms smoothing
    envelope = np.convolve(envelope, np.ones(k) / k, mode="same")

    threshold = np.mean(envelope) + threshold_sd * np.std(envelope)
    above = (envelope > threshold).astype(int)
    onsets = np.where(np.diff(above) == 1)[0] + 1

    margin_pre = int(abs(tmin) * sfreq)
    margin_post = int(tmax * sfreq)
    onsets = onsets[(onsets >= margin_pre) & (onsets < len(neck_ref) - margin_post)]

    if len(onsets) > 1:
        min_gap = int(0.2 * sfreq)
        keep = np.concatenate([[True], np.diff(onsets) >= min_gap])
        onsets = onsets[keep]

    if len(onsets) == 0:
        return np.empty((0, 3), dtype=int)

    return np.column_stack(
        [onsets, np.zeros(len(onsets), int), np.ones(len(onsets), int)]
    )


# %% EMG correction — diagnostics


def _plot_emg_results(
    model,
    sources,
    neck_ref,
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
    """Diagnostic plots for DSS-based EMG removal. Returns index of best component.

    Mirrors plot_eog_results from EOG_correction but uses neck_ref instead of EOG.
    """

    def _save(fig, plot_name):
        if out_dir is not None:
            _save_fig(fig, out_dir, entities, desc=f"{desc}-{plot_name}")

    # component diagnostics
    try:
        fig = plot_component_score_curve(model, mode="ratio", show=False)
        _save(fig, "scoreCurve")
    except ValueError:
        print("Score curve not available for this model type: skipping.")

    if data is not None:
        fig = plot_component_time_series(
            model, data=data, n_components=n_components, show=False
        )
        _save(fig, "timeSeries")

    fig = plot_component_patterns(
        model, info=info, picks=picks, n_components=n_components, show=False
    )
    _save(fig, "patterns")

    if data is not None:
        fig = plot_component_summary(
            model,
            data=data,
            info=info,
            picks=picks,
            n_components=list(range(min(4, n_components))),
            show=False,
        )
        _save(fig, "summary")

    # candidate components: those with above-average score
    try:
        scores = _get_scores(model)
        mean_score = np.mean(scores)
        candidates = [i for i, s in enumerate(scores) if s >= mean_score]
    except Exception:
        candidates = list(range(sources.shape[0]))

    # select component most correlated with neck reference
    n_common = min(sources.shape[1], len(neck_ref))
    nref = neck_ref[:n_common]
    corrs = np.array(
        [abs(np.corrcoef(nref, sources[i, :n_common])[0, 1]) for i in candidates]
    )
    for i, c in zip(candidates, corrs):
        print(f"  Comp {i}: |r| with neck ref = {c:.3f}")
    best_local = int(np.argmax(corrs))
    best_idx = candidates[best_local]
    best_corr = corrs[best_local]
    print(f"→ Removing Comp {best_idx} (neck corr={best_corr:.3f})")

    # neck ref vs best component overlay
    end_idx = min(end_idx, sources.shape[1], len(neck_ref))
    t_window = np.arange(start_idx, end_idx) / info["sfreq"]
    nref_snippet = neck_ref[start_idx:end_idx]
    comp_snippet = sources[best_idx, start_idx:end_idx]
    flip = -1 if np.corrcoef(nref_snippet, comp_snippet)[0, 1] < 0 else 1

    fig, (ax_ref, ax_comp) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        f"{title_prefix}: neck ref vs Comp {best_idx}  (|r|={best_corr:.3f})",
        fontsize=11,
    )
    ax_ref.plot(t_window, nref_snippet, color="steelblue", linewidth=1.5)
    ax_ref.set_ylabel("neck PCA ref (a.u.)")
    ax_ref.grid(True, alpha=0.3)
    ax_ref.axhline(0, color="gray", linewidth=0.6)
    ax_comp.plot(t_window, flip * comp_snippet, color="tomato", linewidth=1.5)
    ax_comp.set_ylabel(f"Comp {best_idx} (a.u.{', flipped' if flip == -1 else ''})")
    ax_comp.set_xlabel("Time (s)")
    ax_comp.grid(True, alpha=0.3)
    ax_comp.axhline(0, color="gray", linewidth=0.6)
    _save(fig, "emgOverlay")

    return best_idx


# %% EMG correction — methods


def linear_dss_EMG_removal(raw_eeg, neck_ref, out_dir, entities=None):
    """Average-bias DSS on burst-locked epochs.

    Mirrors linear_dss_EOG_removal: blink epochs → AverageBias ≡ EMG burst epochs →
    AverageBias. The DSS component most correlated with the neck PCA reference
    is removed.

    Returns
    -------
    raw_clean : mne.io.Raw
        Head EEG with dominant EMG DSS component removed.
    """
    sfreq = raw_eeg.info["sfreq"]
    events = _make_burst_events(neck_ref, sfreq)
    if len(events) == 0:
        warnings.warn(
            "linear_dss_EMG_removal: no EMG bursts detected; returning unchanged."
        )
        return raw_eeg.copy()

    # embed neck_ref as a misc channel so epochs are locked to the artifact source,
    # mirroring create_eog_epochs which epochs the full raw (incl. EOG) then picks MEG
    info_neck = mne.create_info(["neck_ref"], sfreq, ch_types="misc")
    raw_neck_ch = mne.io.RawArray(neck_ref[np.newaxis, :], info_neck, verbose=False)
    raw_combined = raw_eeg.copy().add_channels([raw_neck_ch], force_update_info=True)

    emg_epochs = mne.Epochs(
        raw_combined,
        events,
        event_id={"emg_burst": 1},
        tmin=-0.3,
        tmax=0.3,
        baseline=(-0.3, -0.15),
        preload=True,
        verbose=False,
    )
    emg_epochs.pick("eeg")
    print(f"Linear DSS: {len(events)} bursts detected, {len(emg_epochs)} epochs kept.")

    if len(emg_epochs) < 5:
        warnings.warn("Too few EMG epochs (<5); returning unchanged.")
        return raw_eeg.copy()

    raw_picks = np.arange(len(raw_eeg.ch_names))
    dss_emg = DSS(
        n_components=10, bias=AverageBias(axis="epochs"), return_type="sources"
    )
    dss_emg.fit(emg_epochs)
    sources = dss_emg.transform(raw_eeg)

    best_idx = _plot_emg_results(
        model=dss_emg,
        sources=sources,
        neck_ref=neck_ref,
        info=emg_epochs.info,
        picks=raw_picks,
        data=emg_epochs,
        title_prefix="AverageBias (linear)",
        n_components=10,
        out_dir=out_dir,
        entities=entities,
        desc="linearDSS",
    )

    keep_idx = [i for i in range(sources.shape[0]) if i != best_idx]
    raw_clean = raw_eeg.copy()
    raw_clean._data = dss_emg.inverse_transform(sources, component_indices=keep_idx)
    return raw_clean


def nonlinear_dss_EMG_removal(raw_eeg, neck_ref, out_dir, entities=None):
    """Kurtosis-based IterativeDSS on head EEG.

    Unsupervised: no EMG reference needed for fitting. The neck_ref is used only
    to select which kurtotic component to remove (same role as EOG correlation in
    the nonlinear EOG removal).

    Returns
    -------
    raw_clean : mne.io.Raw
    """
    data = raw_eeg.get_data()
    raw_picks = np.arange(len(raw_eeg.ch_names))

    denoiser = KurtosisDenoiser(nonlinearity="tanh")
    it_dss = IterativeDSS(denoiser, n_components=5, max_iter=100)
    it_dss.fit(data)
    sources = it_dss.transform(data)

    best_idx = _plot_emg_results(
        model=it_dss,
        sources=sources,
        neck_ref=neck_ref,
        info=raw_eeg.info,
        picks=raw_picks,
        data=data,
        title_prefix="KurtosisDSS (nonlinear)",
        n_components=5,
        out_dir=out_dir,
        entities=entities,
        desc="nonlinearDSS",
    )

    sources_clean = sources.copy()
    sources_clean[best_idx, :] = 0
    raw_clean = raw_eeg.copy()
    raw_clean._data = it_dss.inverse_transform(sources_clean)
    return raw_clean


def ica_head_EMG_removal(raw_eeg, rng_seed=None):
    """Fresh ICA on head-only EEG; ICLabel excludes muscle-artifact components.

    Returns
    -------
    raw_clean : mne.io.Raw
    ica : mne.preprocessing.ICA
    ic_labels : dict
    """
    ica, ic_labels = compute_ica(
        raw_eeg.copy().pick("eeg"),
        filter_bands_ica=(1.0, 100.0),
        notch_freqs=(50, 100, 150),
        downsample_ica=250,
        thresh=0.7,
        rng_seed=rng_seed,
        exclude_labels=["muscle artifact"],
    )
    for idx, (lbl, prob) in enumerate(
        zip(ic_labels["labels"], ic_labels["y_pred_proba"])
    ):
        marker = " ← excluded" if idx in ica.exclude else ""
        print(f"  IC{idx:03d}: {lbl:<22s} prob={prob:.3f}{marker}")
    print(f"ICA-head: excluding {len(ica.exclude)} component(s): {ica.exclude}")

    raw_clean = ica.apply(raw_eeg.copy().pick("eeg"))
    return raw_clean, ica, ic_labels


def ica_neck_EMG_removal(raw_eeg, raw_neck, neck_ref, rng_seed=None):
    """ICA fitted on combined head+neck channels; ICs excluded by neck-ref correlation.

    Including neck channels gives the ICA more information to separate scalp EMG from
    brain sources. Component exclusion is driven by the neck PCA reference (not ICLabel,
    since neck channels are not in the standard 10-20 montage).

    Returns
    -------
    raw_clean : mne.io.Raw
        Head-EEG with EMG components removed.
    ica : mne.preprocessing.ICA
        ICA fitted on combined data.
    neck_corrs : ndarray
        Absolute Pearson |r| between each IC and the neck reference.
    n_head : int
        Number of head channels (= first n_head rows of ica.get_components()).
    """
    n_head = len(raw_eeg.ch_names)

    # temporarily retype neck channels as eeg for joint decomposition
    raw_neck_eeg = raw_neck.copy()
    raw_neck_eeg.set_channel_types({ch: "eeg" for ch in raw_neck_eeg.ch_names})

    raw_combined = raw_eeg.copy()
    raw_combined.add_channels([raw_neck_eeg], force_update_info=True)

    # filter / resample / reference (bypass compute_ica to avoid
    # FASTER on non-standard chs)
    raw_filt = raw_combined.copy()
    raw_filt.filter(1.0, None, verbose=False)
    raw_filt.filter(None, 100.0, verbose=False)
    raw_filt.notch_filter([50, 100, 150], notch_widths=1.0, verbose=False)
    if raw_filt.info["sfreq"] > 250:
        raw_filt.resample(250, verbose=False)
    raw_filt.set_eeg_reference("average", verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw_filt, duration=1.0, preload=True, reject_by_annotation=True, verbose=False
    )

    ica = mne.preprocessing.ICA(
        n_components=None,
        random_state=rng_seed,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
    )
    ica.fit(epochs, verbose=False)

    # get sources from combined raw at original sfreq
    raw_for_sources = raw_combined.copy().set_eeg_reference("average", verbose=False)
    sources = ica.get_sources(raw_for_sources).get_data()  # (n_ics, n_times)

    n_common = min(sources.shape[1], len(neck_ref))
    nref = neck_ref[:n_common]
    neck_corrs = np.array(
        [
            abs(np.corrcoef(nref, sources[i, :n_common])[0, 1])
            for i in range(sources.shape[0])
        ]
    )

    above_thresh = np.where(neck_corrs >= NECK_CORR_THRESHOLD)[0].tolist()
    exclude_idx = above_thresh if above_thresh else [int(np.argmax(neck_corrs))]
    ica.exclude = exclude_idx
    print(f"ICA+neck: excluding {len(exclude_idx)} IC(s): {exclude_idx}")
    for i in exclude_idx:
        print(f"  IC{i:03d}: neck_corr={neck_corrs[i]:.3f}")

    raw_clean_combined = ica.apply(raw_for_sources.copy())
    raw_clean = raw_clean_combined.copy().pick(raw_eeg.ch_names)
    return raw_clean, ica, neck_corrs, n_head


# %% Metrics


def compute_emg_amplitudes(
    raw_original, cleaned_raws, neck_ref, sfreq, tmin=-0.3, tmax=0.3
):
    """Per-burst amplitude at the peak EMG channel/latency for each recording.

    Analogous to compute_blink_amplitudes in EOG_correction. Burst onsets are
    detected from the neck reference; the peak (channel, latency) coordinate is
    determined blindly from the original (uncleaned) average.

    Parameters
    ----------
    raw_original : mne.io.Raw
        Uncleaned head EEG (used to locate the peak artifact coordinate).
    cleaned_raws : dict[str, mne.io.Raw]
        Method label → cleaned raw.
    neck_ref : ndarray, shape (n_times,)
    sfreq : float
    tmin, tmax : float
        Burst epoch window.

    Returns
    -------
    df : pd.DataFrame
        Shape (n_bursts, 1 + n_methods). Columns: ``"original"`` + method keys.
        Values are amplitude in µV at the peak (channel, latency) coordinate.
    peak_info : dict
        ``{"channel": str, "latency_s": float}``.
    """
    events = _make_burst_events(neck_ref, sfreq, tmin=tmin, tmax=tmax)
    if len(events) == 0:
        warnings.warn("compute_emg_amplitudes: no burst events found.")
        return pd.DataFrame(), {}

    all_raws = {"original": raw_original} | cleaned_raws
    epochs_dict = {}
    for label, rw in all_raws.items():
        ep = mne.Epochs(
            rw,
            events,
            event_id={"emg_burst": 1},
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        ep.pick("eeg")
        epochs_dict[label] = ep

    # peak coordinate from original average
    orig_avg = epochs_dict["original"].average().data  # (n_ch, n_times)
    best_ch_idx = int(np.argmax(np.abs(orig_avg).max(axis=1)))
    best_t_idx = int(np.argmax(np.abs(orig_avg[best_ch_idx])))
    ch_name = epochs_dict["original"].ch_names[best_ch_idx]
    peak_latency_s = float(epochs_dict["original"].times[best_t_idx])

    records = {
        label: ep.get_data()[:, best_ch_idx, best_t_idx] * 1e6
        for label, ep in epochs_dict.items()
    }
    return pd.DataFrame(records), {"channel": ch_name, "latency_s": peak_latency_s}


def compute_residual_neck_corr(raw_eeg, neck_ref):
    """Mean absolute Pearson |r| between cleaned EEG channels and neck PCA reference.

    Lower values indicate less residual EMG contamination.
    """
    data = raw_eeg.get_data(picks="eeg")  # (n_ch, n_times)
    n_common = min(data.shape[1], len(neck_ref))
    nref = neck_ref[:n_common]
    corrs = np.array(
        [abs(np.corrcoef(data[i, :n_common], nref)[0, 1]) for i in range(data.shape[0])]
    )
    return float(np.mean(corrs))


def compute_lowfreq_preservation(raw_before, raw_after, l_freq=None, h_freq=None):
    """Power ratio in the alpha/beta band after vs before cleaning (target ≈ 1).

    Values below 1 indicate that low-frequency EEG was attenuated (over-cleaning).
    """
    l_freq = l_freq or ALPHA_BETA_BAND[0]
    h_freq = h_freq or ALPHA_BETA_BAND[1]
    sfreq = raw_before.info["sfreq"]
    nperseg = int(sfreq * 4)

    def _band_power(rw):
        psd_data = rw.get_data(picks="eeg")
        freqs, psd = welch(psd_data, fs=sfreq, nperseg=nperseg, axis=1)
        band = (freqs >= l_freq) & (freqs <= h_freq)
        return float(np.mean(psd[:, band]))

    p_before = _band_power(raw_before)
    p_after = _band_power(raw_after)
    return p_after / (p_before + 1e-12)


def _fit_post_dss_ica(raw_clean, rng_seed=None):
    """Fit ICA on DSS-cleaned head EEG without label exclusion (for dipolarity only)."""
    ica, ic_labels = compute_ica(
        raw_clean.copy().pick("eeg"),
        filter_bands_ica=(1.0, 100.0),
        notch_freqs=(50, 100, 150),
        downsample_ica=250,
        thresh=0.7,
        rng_seed=rng_seed,
        exclude_labels=None,
        include_labels=None,
    )
    return ica, ic_labels


def compute_all_metrics(
    raw_preproc, emg_cleaned, method_icas, neck_ref, head_info, rng_seed=None
):
    """Compute all objective metrics for each EMG correction method.

    Parameters
    ----------
    raw_preproc : mne.io.Raw
        Preprocessed (but not EMG-corrected) head EEG — the shared baseline.
    emg_cleaned : dict[str, mne.io.Raw]
        Method label → cleaned head EEG.
    method_icas : dict[str, tuple | None]
        Method label → (ica, n_head) or None.
        - DSS methods: None → fresh ICA fitted post-cleaning for dipolarity.
        - ICA-head: (ica, None) → use retained components directly.
        - ICA-neck: (ica, n_head) → slice components to head channels.
    neck_ref : ndarray, shape (n_times,)
    head_info : mne.Info
        Info for the head-only EEG channels (used for dipolarity).
    rng_seed : int | None

    Returns
    -------
    pd.DataFrame
        One row per method, columns: method + all metric names.
    dict[str, dict]
        Per-method dipolarity details (``rv_values`` etc.).
    """
    rows = []
    dip_details = {}

    for method, raw_clean in emg_cleaned.items():
        print(f"\n--- Metrics: {method} ---")
        row = {"method": method}

        # --- mutual information reduction ---
        mi = compute_mi_reduction(raw_preproc, raw_clean)
        row["mi_reduction"] = mi["mi_reduction"]
        row["mi_reduction_pct"] = mi["mi_reduction_pct"]

        # --- residual neck correlation ---
        row["residual_neck_corr"] = compute_residual_neck_corr(raw_clean, neck_ref)

        # --- low-frequency preservation ---
        row["lowfreq_preservation"] = compute_lowfreq_preservation(
            raw_preproc, raw_clean
        )

        # --- dipolarity ---
        ica_info = method_icas.get(method)
        if ica_info is None:
            # DSS method: fit fresh ICA on cleaned data
            print(f"  Fitting post-DSS ICA for dipolarity ({method})...")
            ica_for_dip, _ = _fit_post_dss_ica(raw_clean, rng_seed=rng_seed)
            components = ica_for_dip.get_components()
            info_for_dip = head_info
        else:
            ica, n_head = ica_info
            if n_head is not None:
                # ICA-neck: use head-channel rows of the combined mixing matrix
                components = ica.get_components()[:n_head, :]
                info_for_dip = head_info
            else:
                # ICA-head: use retained components only
                retained = [i for i in range(ica.n_components_) if i not in ica.exclude]
                components = ica.get_components()[:, retained]
                info_for_dip = head_info

        dip = compute_dipolarity(components, info_for_dip, rv_thresh=RV_THRESHOLD)
        row["fraction_dipolar"] = dip["fraction_dipolar"]
        row["n_dipolar"] = dip["n_dipolar"]
        dip_details[method] = dip

        print(
            f"  MI reduction: {row['mi_reduction']:.3f}"
            f" ({row['mi_reduction_pct']:.1f}%)"
        )
        print(f"  Residual neck corr: {row['residual_neck_corr']:.4f}")
        print(f"  Low-freq preservation: {row['lowfreq_preservation']:.4f}")
        print(
            f"  Dipolarity: {row['fraction_dipolar']:.3f} ({row['n_dipolar']} / "
            f"{len(dip['rv_values'])} ICs)"
        )
        rows.append(row)

    return pd.DataFrame(rows), dip_details


# %% Save


def save_bids_derivatives(
    entities,
    raw_minimal,
    raw_clean,
    emg_cleaned,
    ica,
    ic_labels,
    bad_ch_dict,
    emg_df,
    peak_info,
    metrics_df,
    *,
    overwrite=False,
):
    """Save all per-subject derivatives in BIDS layout.

    Output layout::

        DERIV_DIR / PIPELINE_NAME / sub-XX / [ses-XX /] eeg /
            *_desc-minimal_eeg.fif.gz
            *_desc-preproc-clean_eeg.fif.gz
            *_desc-linearDSS_eeg.fif.gz
            *_desc-nonlinearDSS_eeg.fif.gz
            *_desc-icaHead_eeg.fif.gz
            *_desc-icaNeck_eeg.fif.gz
            *_desc-muscleICA_ica.fif.gz
            *_iclabels.json
            *_bads.json
            *_desc-emgAmplitudes_channels.tsv
            *_desc-emgAmplitudes_channels.json
            *_desc-metrics_channels.tsv
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
    for key, rw in emg_cleaned.items():
        rw.save(fpath(desc=_EMG_DESC[key]), overwrite=overwrite)

    ica.save(fpath(desc="muscleICA", suffix="ica"), overwrite=overwrite)

    with open(fpath(suffix="iclabels", ext=".json"), "w") as f:
        json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)
    with open(fpath(suffix="bads", ext=".json"), "w") as f:
        json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)

    tsv_path = fpath(desc="emgAmplitudes", suffix="channels", ext=".tsv")
    if overwrite or not tsv_path.exists():
        emg_df.to_csv(tsv_path, sep="\t", index=False)
    with open(fpath(desc="emgAmplitudes", suffix="channels", ext=".json"), "w") as f:
        json.dump(peak_info, f, indent=4)

    metrics_path = fpath(desc="metrics", suffix="channels", ext=".tsv")
    if overwrite or not metrics_path.exists():
        metrics_df.to_csv(metrics_path, sep="\t", index=False)


# %% Plotting


def plot_emg_erp_comparison(data, out_path):
    """Burst-locked ERP comparison across methods. Mirrors plot_blink_erp_comparison."""
    colors = ["k", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    is_group = isinstance(next(iter(data.values())), list)
    stats = {}
    for label, val in data.items():
        if is_group:
            grand = mne.grand_average(val)
            ch_names = grand.ch_names
            times = grand.times
            mean_uv = grand.data * 1e6
            sub_mat = (
                np.stack([e.copy().pick_channels(list(ch_names)).data for e in val])
                * 1e6
            )
            sem_uv = sub_mat.std(axis=0) / np.sqrt(len(val))
        else:
            ch_names = val.ch_names
            times = val.times
            mean_uv = val.average().data * 1e6
            sem_uv = val.get_data().std(axis=0) / np.sqrt(len(val)) * 1e6
        stats[label] = (mean_uv, sem_uv, times, list(ch_names))

    orig_label = "original" if "original" in stats else next(iter(stats))
    orig_mean, _, times, orig_chs = stats[orig_label]
    best_ch_idx = int(np.argmax(np.abs(orig_mean).max(axis=1)))
    ch_name = orig_chs[best_ch_idx]

    _, ax = plt.subplots(figsize=(10, 4))
    for (label, (mean_uv, sem_uv, _, ch_names)), color in zip(stats.items(), colors):
        idx = ch_names.index(ch_name)
        ax.plot(times, mean_uv[idx], color=color, linewidth=1.8, label=label)
        ax.fill_between(
            times,
            mean_uv[idx] - sem_uv[idx],
            mean_uv[idx] + sem_uv[idx],
            color=color,
            alpha=0.15,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    prefix = "Group " if is_group else ""
    ax.set_title(
        f"{prefix}burst-locked ERP: {ch_name}  (residual = artefact remaining)"
    )
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_emg_amplitude_comparison(df, out_path):
    """Violin plot of per-burst EMG amplitudes across methods."""
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

    axes[0][0].set_ylabel("amplitude at EMG peak (µV)")
    prefix = "group " if is_group else ""
    fig.suptitle(f"{prefix}burst amplitude: residual per method", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def plot_group_emg_amplitude(group_df, out_path):
    """Violin + paired subject-mean scatter for group burst-amplitude comparison."""
    method_cols = [c for c in group_df.columns if c != "sub"]
    subs = sorted(group_df["sub"].unique())
    n_methods = len(method_cols)

    sub_means = (
        group_df.groupby("sub")[method_cols]
        .apply(lambda g: g.abs().mean())
        .reindex(subs)
    )

    fig, ax = plt.subplots(figsize=(max(5, 2 * n_methods + 1), 5))
    positions = list(range(1, n_methods + 1))
    vals = [group_df[col].dropna().abs().values for col in method_cols]
    parts = ax.violinplot(
        vals, positions=positions, showmedians=True, showextrema=False
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for sub in subs:
        row = sub_means.loc[sub]
        x = [p + rng.uniform(-0.08, 0.08) for p in positions]
        y = [row[col] for col in method_cols]
        ax.plot(x, y, color="steelblue", alpha=0.4, linewidth=0.8, zorder=2)
        ax.scatter(x, y, s=22, color="steelblue", alpha=0.7, zorder=3)

    group_means = [sub_means[col].mean() for col in method_cols]
    ax.scatter(
        positions,
        group_means,
        s=60,
        color="k",
        zorder=4,
        marker="D",
        label="group mean",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(method_cols, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("|amplitude| at EMG peak (µV)")
    ax.set_title("burst residual amplitude by EMG-correction method", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def plot_metrics_comparison(metrics_df, out_path):
    """Bar chart comparing all scalar metrics across the four EMG-correction methods.

    One subplot per metric, one bar per method.
    """
    metric_cols = [c for c in metrics_df.columns if c != "method"]
    metric_labels = {
        "mi_reduction": "MI reduction (nats)",
        "mi_reduction_pct": "MI reduction (%)",
        "residual_neck_corr": "residual neck corr (|r|)",
        "lowfreq_preservation": "low-freq preservation (ratio)",
        "fraction_dipolar": "fraction dipolar ICs",
        "n_dipolar": "# dipolar ICs",
    }

    cols_to_plot = [c for c in metric_cols if c in metric_labels]
    if not cols_to_plot:
        return

    n = len(cols_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    methods = metrics_df["method"].tolist()

    for ax, col in zip(axes, cols_to_plot):
        vals = metrics_df[col].values
        _ = ax.bar(range(len(methods)), vals, color=colors[: len(methods)], alpha=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.set_title(metric_labels.get(col, col), fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        # reference line for preservation (target = 1)
        if col == "lowfreq_preservation":
            ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)

    fig.suptitle("Objective metrics per EMG-correction method", fontsize=11)
    fig.savefig(out_path, dpi=300)
    plt.close()


def plot_group_metrics(group_metrics_df, out_path):
    """Violin + paired subject scatter for each metric across methods (group level)."""
    metric_labels = {
        "mi_reduction": "MI reduction (nats)",
        "mi_reduction_pct": "MI reduction (%)",
        "residual_neck_corr": "residual neck corr (|r|)",
        "lowfreq_preservation": "low-freq preservation",
        "fraction_dipolar": "fraction dipolar ICs",
    }
    non_metric = {"sub", "method"}
    metric_cols = [
        c
        for c in group_metrics_df.columns
        if c not in non_metric and c in metric_labels
    ]
    methods = sorted(group_metrics_df["method"].unique(), key=list(_EMG_DESC).index)
    subs = sorted(group_metrics_df["sub"].unique())
    n_metrics = len(metric_cols)

    fig, axes = plt.subplots(
        1, n_metrics, figsize=(3.5 * n_metrics, 5), constrained_layout=True
    )
    if n_metrics == 1:
        axes = [axes]

    rng = np.random.default_rng(1)
    for ax, col in zip(axes, metric_cols):
        positions = list(range(1, len(methods) + 1))
        vals = [
            group_metrics_df.loc[group_metrics_df["method"] == m, col].dropna().values
            for m in methods
        ]
        if any(len(v) > 1 for v in vals):
            parts = ax.violinplot(
                vals, positions=positions, showmedians=True, showextrema=False
            )
            for pc in parts["bodies"]:
                pc.set_alpha(0.35)

        for sub in subs:
            sub_vals = [
                group_metrics_df.loc[
                    (group_metrics_df["sub"] == sub)
                    & (group_metrics_df["method"] == m),
                    col,
                ].values
                for m in methods
            ]
            x = [p + rng.uniform(-0.06, 0.06) for p in positions]
            y = [v[0] if len(v) == 1 else np.nan for v in sub_vals]
            ax.plot(x, y, color="steelblue", alpha=0.4, linewidth=0.8)
            ax.scatter(x, y, s=20, color="steelblue", alpha=0.7, zorder=3)

        group_means = [np.nanmean(v) for v in vals]
        ax.scatter(positions, group_means, s=50, color="k", zorder=4, marker="D")
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.set_title(metric_labels[col], fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        if col == "lowfreq_preservation":
            ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)

    fig.suptitle("Group metrics by EMG-correction method", fontsize=11)
    fig.savefig(out_path, dpi=300)
    plt.close()


# %% Driver helpers


def _expected_outputs_bids(entities: dict) -> list[Path]:
    out_dir = DERIV_DIR / PIPELINE_NAME / f"sub-{entities['sub']}"
    if "ses" in entities:
        out_dir = out_dir / f"ses-{entities['ses']}"
    out_dir = out_dir / "eeg"

    def fpath(desc=None, suffix="eeg", ext=".fif.gz"):
        return out_dir / (_bids_stem(entities, desc=desc, suffix=suffix) + ext)

    return [
        fpath(desc="minimal"),
        fpath(desc="preproc-clean"),
        *(fpath(desc=desc) for desc in _EMG_DESC.values()),
        fpath(desc="muscleICA", suffix="ica"),
        fpath(desc="emgERP-comparison", suffix="fig", ext=".png"),
        fpath(desc="emgAmplitude-comparison", suffix="fig", ext=".png"),
        fpath(desc="metrics-comparison", suffix="fig", ext=".png"),
    ]


def classify_responders(summary_df, threshold_pct=RESPONDER_THRESHOLD_PCT):
    """Classify subjects as responders per method (>= threshold_pct % RMS reduction)."""
    rows = []
    for _, row in summary_df.iterrows():
        orig_rms = row["original_rms"]
        for method in _EMG_DESC:
            method_rms = row.get(f"{method}_rms", np.nan)
            if orig_rms > 0 and not np.isnan(method_rms):
                pct = (orig_rms - method_rms) / orig_rms * 100.0
            else:
                pct = np.nan
            rows.append(
                {
                    "sub": row["sub"],
                    "method": method,
                    "original_rms": orig_rms,
                    "method_rms": method_rms,
                    "pct_reduction": pct,
                    "responder": bool(pct >= threshold_pct)
                    if not np.isnan(pct)
                    else False,
                }
            )
    return pd.DataFrame(rows)


# %% Main single-subject pipeline


def run_single_subject(bids_path: BIDSPath):
    """Run the full EMG-correction comparison pipeline for one BIDS recording.

    Pipeline
    --------
    1. Read BIDS file; separate head EEG from neck channels.
    2. Shared preprocessing (PREPROCESSOR) on head EEG.
    3. Compute neck PCA reference (PC with highest kurtosis).
    4. Four EMG correction methods on ``raw_clean``:
       - Linear DSS (burst-epoch AverageBias, neck-ref–guided component selection)
       - Nonlinear DSS (KurtosisDenoiser, neck-ref–guided component selection)
       - ICA head-only (ICLabel muscle exclusion)
       - ICA head+neck (neck-ref–guided IC exclusion)
    5. All objective metrics per method.
    6. Save BIDS derivatives.
    7. Burst-locked ERP and amplitude plots.
    """
    entities = _entities_from_bids_path(bids_path)
    label = bids_path.fpath.name

    if not FORCE_RERUN and all(p.exists() for p in _expected_outputs_bids(entities)):
        print(f"[SKIP] {label}: all outputs present")
        return

    print(f"\n=== Processing {label} ===")
    raw = _load_bids_raw(bids_path)

    neck_chs = _detect_neck_channels(raw)
    if not neck_chs:
        print(
            f"[SKIP] {label}: no neck channels found (prefix='{NECK_CHANNEL_PREFIX}')"
        )
        return
    print(
        f"Detected {len(neck_chs)} neck channel(s): {neck_chs[:4]}"
        f"{'...' if len(neck_chs) > 4 else ''}"
    )
    sub_dir = DERIV_DIR / PIPELINE_NAME / f"sub-{entities['sub']}"
    if "ses" in entities:
        sub_dir = sub_dir / f"ses-{entities['ses']}"
    sub_dir = sub_dir / "eeg"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # separate head EEG and neck channels
    raw_neck = raw.copy().pick_channels(neck_chs)
    raw.drop_channels(neck_chs)

    # shared preprocessing on head EEG only
    raw_minimal, raw_clean, _, ica_preproc, ic_labels_preproc, bad_ch_dict = (
        PREPROCESSOR.run_raw(raw)
    )

    # propagate BAD annotations
    bad_mask = np.array(
        [d.upper().startswith("BAD") for d in raw_minimal.annotations.description]
    )
    bad_annots = mne.Annotations(
        raw_minimal.annotations.onset[bad_mask],
        raw_minimal.annotations.duration[bad_mask],
        raw_minimal.annotations.description[bad_mask],
        orig_time=raw_minimal.annotations.orig_time,
    )
    for rw in (raw_minimal, raw_clean):
        rw.set_annotations(bad_annots)

    # align neck data to preprocessed sfreq (resampled by PREPROCESSOR)
    if raw_neck.info["sfreq"] != raw_clean.info["sfreq"]:
        raw_neck.resample(raw_clean.info["sfreq"], verbose=False)

    # neck PCA reference
    neck_data = raw_neck.get_data()
    neck_ref, pca_info = compute_neck_pca_reference(neck_data)

    with open(
        sub_dir / (_bids_stem(entities, desc="neckPCA", suffix="channels") + ".json"),
        "w",
    ) as f:
        json.dump(pca_info, f, indent=4, cls=NumpyEncoder)

    # --- four EMG correction methods ---
    emg_cleaned = {}
    method_icas = {}

    # linear DSS
    emg_cleaned["linear_dss"] = linear_dss_EMG_removal(
        raw_clean, neck_ref, out_dir=sub_dir, entities=entities
    )
    method_icas["linear_dss"] = None  # post-DSS ICA fitted in compute_all_metrics

    # nonlinear DSS
    emg_cleaned["nonlinear_dss"] = nonlinear_dss_EMG_removal(
        raw_clean, neck_ref, out_dir=sub_dir, entities=entities
    )
    method_icas["nonlinear_dss"] = None

    # ICA head-only
    raw_ica_head, ica_head, ic_labels_head = ica_head_EMG_removal(
        raw_clean, rng_seed=PREPROCESSOR.rng_seed
    )
    emg_cleaned["ica_head"] = raw_ica_head
    method_icas["ica_head"] = (ica_head, None)

    # ICA head+neck
    raw_ica_neck, ica_neck, neck_corrs, n_head = ica_neck_EMG_removal(
        raw_clean, raw_neck, neck_ref, rng_seed=PREPROCESSOR.rng_seed
    )
    emg_cleaned["ica_neck"] = raw_ica_neck
    method_icas["ica_neck"] = (ica_neck, n_head)

    # save ICA objects for reference
    ica_head.save(
        sub_dir / (_bids_stem(entities, desc="icaHeadEMG", suffix="ica") + ".fif.gz"),
        overwrite=FORCE_RERUN,
    )
    ica_neck.save(
        sub_dir / (_bids_stem(entities, desc="icaNeckEMG", suffix="ica") + ".fif.gz"),
        overwrite=FORCE_RERUN,
    )

    # --- objective metrics ---
    metrics_df, dip_details = compute_all_metrics(
        raw_clean,
        emg_cleaned,
        method_icas,
        neck_ref,
        head_info=raw_clean.info,
        rng_seed=PREPROCESSOR.rng_seed,
    )

    # save per-method dipolarity details
    with open(
        sub_dir
        / (_bids_stem(entities, desc="dipolarity", suffix="channels") + ".json"),
        "w",
    ) as f:
        json.dump(
            {
                m: {k: v for k, v in d.items() if k != "rv_values"}
                | {"rv_values": d["rv_values"]}
                for m, d in dip_details.items()
            },
            f,
            indent=4,
            cls=NumpyEncoder,
        )

    # --- burst-locked amplitude table ---
    sfreq = raw_clean.info["sfreq"]
    emg_df, peak_info = compute_emg_amplitudes(
        raw_minimal, emg_cleaned, neck_ref, sfreq
    )

    # --- save all derivatives ---
    save_bids_derivatives(
        entities,
        raw_minimal,
        raw_clean,
        emg_cleaned,
        ica_preproc,
        ic_labels_preproc,
        bad_ch_dict,
        emg_df,
        peak_info,
        metrics_df,
        overwrite=FORCE_RERUN,
    )

    # --- burst-locked ERP epochs for plot ---
    events = _make_burst_events(neck_ref, sfreq)
    erp_data = {}
    for label, rw in ({"original": raw_minimal} | emg_cleaned).items():
        if len(events) == 0:
            break
        ep = mne.Epochs(
            rw,
            events,
            event_id={"emg_burst": 1},
            tmin=-0.3,
            tmax=0.3,
            baseline=None,
            preload=True,
            verbose=False,
        )
        ep.pick("eeg")
        erp_data[label] = ep

    if erp_data:
        plot_emg_erp_comparison(
            erp_data,
            out_path=sub_dir
            / (_bids_stem(entities, desc="emgERP-comparison", suffix="fig") + ".png"),
        )

    if not emg_df.empty:
        plot_emg_amplitude_comparison(
            emg_df,
            out_path=sub_dir
            / (
                _bids_stem(entities, desc="emgAmplitude-comparison", suffix="fig")
                + ".png"
            ),
        )

    plot_metrics_comparison(
        metrics_df,
        out_path=sub_dir
        / (_bids_stem(entities, desc="metrics-comparison", suffix="fig") + ".png"),
    )

    clear_matplotlib_memory()


# %% Group pipeline


def run_group():
    """Load per-subject TSVs and produce group-level amplitude and metrics reports."""
    out_dir = DERIV_DIR / PIPELINE_NAME

    # --- amplitude aggregation ---
    amp_files = sorted(
        f
        for f in out_dir.rglob("*_desc-emgAmplitudes_channels.tsv")
        if "sub" in _parse_bids_entities(f)
    )
    if amp_files:
        dfs = []
        for f in amp_files:
            sub_df = pd.read_csv(f, sep="\t")
            sub_df.insert(0, "sub", _parse_bids_entities(f).get("sub", "unknown"))
            dfs.append(sub_df)
        group_df = pd.concat(dfs, ignore_index=True)

        method_cols = [c for c in group_df.columns if c != "sub"]
        orig = group_df["original"].dropna()
        q1, q3 = orig.quantile(0.25), orig.quantile(0.75)
        iqr = q3 - q1
        mask = group_df["original"].between(q1 - 3 * iqr, q3 + 3 * iqr)
        n_dropped = (~mask).sum()
        if n_dropped:
            print(
                f"Dropping {n_dropped} outlier burst trial(s) "
                "(3-IQR fence on original)."
            )
        group_df = group_df[mask].reset_index(drop=True)

        summary_rows = []
        for sub, sub_df in group_df.groupby("sub"):
            row = {"sub": sub}
            for col in method_cols:
                vals = sub_df[col].dropna()
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_rms"] = float(np.sqrt((vals**2).mean()))
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        group_df.to_csv(
            out_dir / "group_desc-emgAmplitudes_channels.tsv", sep="\t", index=False
        )
        summary_df.to_csv(
            out_dir / "group_desc-emgAmplitudesSummary_channels.tsv",
            sep="\t",
            index=False,
        )

        responder_df = classify_responders(summary_df)
        responder_df.to_csv(
            out_dir / "group_desc-responders_channels.tsv", sep="\t", index=False
        )
        n_total = len(summary_df)
        print(
            f"\nResponder summary (threshold: {RESPONDER_THRESHOLD_PCT:.0f}"
            "% RMS reduction):"
        )
        for method, mdf in responder_df.groupby("method"):
            print(f"  {method}: {int(mdf['responder'].sum())}/{n_total} responders")

        plot_group_emg_amplitude(
            group_df, out_path=out_dir / "group_emg-amplitude-comparison.png"
        )

    # --- metrics aggregation ---
    metrics_files = sorted(
        f
        for f in out_dir.rglob("*_desc-metrics_channels.tsv")
        if "sub" in _parse_bids_entities(f)
    )
    if metrics_files:
        mdfs = []
        for f in metrics_files:
            mdf = pd.read_csv(f, sep="\t")
            mdf.insert(0, "sub", _parse_bids_entities(f).get("sub", "unknown"))
            mdfs.append(mdf)
        group_metrics_df = pd.concat(mdfs, ignore_index=True)
        group_metrics_df.to_csv(
            out_dir / "group_desc-metrics_channels.tsv", sep="\t", index=False
        )
        plot_group_metrics(
            group_metrics_df, out_path=out_dir / "group_metrics-comparison.png"
        )

    # --- group ERP ---
    minimal_fifs = sorted(out_dir.rglob("*_desc-minimal_eeg.fif.gz"))
    if minimal_fifs:
        evokeds = {label: [] for label in ["original", *_EMG_DESC]}
        for minimal_fif in minimal_fifs:
            raw_min = mne.io.read_raw_fif(minimal_fif, preload=True, verbose=False)
            neck_ref_candidate = _load_neck_ref_for_group(minimal_fif)
            if neck_ref_candidate is None:
                continue
            events = _make_burst_events(neck_ref_candidate, raw_min.info["sfreq"])
            if len(events) == 0:
                continue

            sub_fifs = {"original": minimal_fif} | {
                key: next(
                    iter(minimal_fif.parent.glob(f"*_desc-{desc}_eeg.fif.gz")), None
                )
                for key, desc in _EMG_DESC.items()
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
                    events,
                    event_id={"emg_burst": 1},
                    tmin=-0.3,
                    tmax=0.3,
                    baseline=None,
                    preload=True,
                    verbose=False,
                )
                ep.pick("eeg")
                if len(ep) > 0:
                    evokeds[label].append(ep.average())

        evokeds = {k: v for k, v in evokeds.items() if v}
        if evokeds:
            plot_emg_erp_comparison(
                evokeds, out_path=out_dir / "group_emgERP-comparison.png"
            )


def _load_neck_ref_for_group(minimal_fif):
    """Load the saved neck PCA info and reconstruct the reference from the minimal FIF.

    Returns None if the neck PCA JSON is not found (subject skipped).
    """
    # find the neck PCA JSON alongside the FIF
    pca_jsons = list(minimal_fif.parent.glob("*_desc-neckPCA_channels.json"))
    if not pca_jsons:
        return None
    with open(pca_jsons[0]) as f:
        pca_info = json.load(f)  # noqa
    # we can't easily reconstruct neck_ref without the raw neck data here;
    # use the burst events stored implicitly in the ERP epochs.
    # For group-level ERP, return None to skip (burst events aren't persisted).
    # To support group ERP fully, save burst events to a TSV in run_single_subject.
    return None  # placeholder — group ERP requires saved event TSVs


# %% Main


def main():
    """Run pipeline for all subjects, then group aggregation."""
    subjects = get_entity_vals(BIDS_ROOT, "subject", ignore_sessions="joy")
    failed = []
    if MODE in ("single", True):
        for sub in sorted(subjects):
            bids_path = BIDSPath(
                subject=sub,
                session=SESSION,
                task=TASK,
                datatype="eeg",
                root=BIDS_ROOT,
            )
            label = f"sub-{sub}_ses-{SESSION}"
            if not bids_path.fpath.exists():
                print(f"[SKIP] {label}: BIDS file not found")
                continue
            if any(skip in label for skip in SKIP_THESE):
                print(f"[SKIP] {label}: on skip list")
                continue
            try:
                run_single_subject(bids_path)
            except Exception as exc:
                print(f"\n[ERROR] {label}: {exc}")
                traceback.print_exc()
                failed.append((label, str(exc)))

    if failed:
        print("\n=== failed subjects ===")
        for name, err in failed:
            print(f"  {name}: {err}")

    if MODE in ("group", True):
        run_group()

    if README_DESTINATION:
        readme_src = Path(__file__).parent / "README.md"
        if readme_src.exists():
            readme_dst = README_DESTINATION
            readme_dst.parent.mkdir(parents=True, exist_ok=True)
            readme_dst.write_text(readme_src.read_text())
            print(f"Saved README to {readme_dst}")


# %% Entry point
if __name__ == "__main__":
    main()

# %%
