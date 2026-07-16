"""Denoiser comparison for HIP data — modular, config-driven.

Edit PIPELINE_CONFIG to enable/disable denoisers, adjust processing parameters,
and supply an optional prehook callable.

Outputs:
  - Per-subject figure  (<deriv>/<subj>/denoise_comparison/*_comparison.png)
    ERP panel: ± SEM across trials (estimation reliability).
  - Per-condition group figure  (<deriv>/group_<cond>_comparison.png)
    ERP panel: ± std across subjects (between-subject variability).

Run AFTER HIP-analysis.py has produced *_preproc_clean.fif.gz files.
"""

# %% Imports

import gc
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_denoise.dss import (
    DSS,
    AverageBias,
    BandpassBias,
    CombFilterBias,
    DCTDenoiser,
    GaussDenoiser,
    IterativeDSS,
    KurtosisDenoiser,
    LineNoiseBias,
    PeakFilterBias,
    QuasiPeriodicDenoiser,
    RobustTanhDenoiser,
    SkewDenoiser,
    SmoothingBias,
    SmoothTanhDenoiser,
    SpectrogramDenoiser,
    TanhMaskDenoiser,
    TimeShiftBias,
    WienerMaskDenoiser,
)
from mne_denoise.zapline import ZapLine
from scipy.signal import welch

# ═══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONFIGURATION — edit here
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(r"C:\Users\random\Documents\Data\young HIP")
DERIV_DIR = DATA_DIR / "derivatives"

REMAPS = {
    # "STaudio": {
    #     "ST_audio/left_second": "incongruent",
    #     "ST_audio/left_first":  "congruent",
    #     "ST_audio/right_second": "congruent",
    #     "ST_audio/right_first":  "incongruent",
    # },
    "STvisual": {
        "ST_visual/left_second": "incongruent",
        "ST_visual/left_first": "congruent",
        "ST_visual/right_second": "congruent",
        "ST_visual/right_first": "incongruent",
    },
    # "DTaudio": { ... },
    "DTvisual": {
        "DT_visual/left_second": "incongruent",
        "DT_visual/left_first": "congruent",
        "DT_visual/right_second": "congruent",
        "DT_visual/right_first": "incongruent",
    },
}

PIPELINE_CONFIG: dict[str, Any] = {
    # Optional callable applied to raw_clean before any denoising.
    # Signature: (raw: mne.io.Raw) -> mne.io.Raw
    # Example: drop extra channels, change reference, etc.
    "prehook": None,
    # Set to True to skip per-subject figures and only produce group plots.
    # Useful when re-running on large cohorts where individual plots are not needed.
    "group_only": True,
    # Subject IDs to exclude entirely (matched against the derivative folder name).
    "exclude_subjects": ["p023"],
    # ERP processing parameters
    "processing": {
        "epoch_tmin": -0.2,
        "epoch_tmax": 1.0,
        "baseline": (-0.2, 0.0),
        "bandpass_erp": (1, 20.0),  # (l_freq, h_freq) applied before epoching
        "line_freq": 50.0,  # displayed as reference marker in PSD
        "erp_channels": ["Cz", "Oz", "Pz", "CPz", "P3", "P4", "FCz"],
    },
    # Denoiser list.  Set enabled=False to skip without removing the entry.
    # "name"  must match a key in DENOISER_REGISTRY below.
    # Optional "label" / "color" override registry defaults for this run.
    "denoisers": [
        # ── Always-on baseline ──────────────────────────────────────────────
        {"name": "baseline", "enabled": True, "params": {}},
        # ── Pre-epoch (raw-stage) denoisers ─────────────────────────────────
        {
            "name": "zapline",
            "enabled": True,
            "params": {"n_remove": "auto", "threshold": 3.0},
        },
        {
            "name": "dss_bandpass",
            "enabled": True,
            "params": {"freq_band": (0.5, 20.0), "n_components": 10},
        },
        {
            "name": "dss_tsr",
            "enabled": True,
            "params": {"shifts": 20, "n_components": 10},
        },
        # ── Post-epoch (epoch-stage) denoisers ──────────────────────────────
        {"name": "dss_average", "enabled": True, "params": {"n_components": 5}},
        # ── Additional raw-stage (disabled by default) ───────────────────────
        {
            "name": "dss_smoothing",
            "enabled": False,
            "params": {"window": 10, "n_components": 10},
        },
        {"name": "dss_linenoise", "enabled": False, "params": {"n_components": 5}},
        {
            "name": "dss_peak",
            "enabled": False,
            "params": {"freq": 10.0, "n_components": 5},
        },
        {
            "name": "dss_comb",
            "enabled": False,
            "params": {"fundamental_freq": 10.0, "n_harmonics": 3, "n_components": 5},
        },
        {
            "name": "dss_kurtosis",
            "enabled": False,
            "params": {"nonlinearity": "tanh", "alpha": 1.0, "n_components": 5},
        },
        {
            "name": "dss_wiener",
            "enabled": False,
            "params": {
                "window_samples": 50,
                "noise_percentile": 25.0,
                "n_components": 5,
            },
        },
        {
            "name": "dss_tanh",
            "enabled": False,
            "params": {"alpha": 1.0, "n_components": 5},
        },
        {
            "name": "dss_robust_tanh",
            "enabled": False,
            "params": {"alpha": 1.0, "n_components": 5},
        },
        {
            "name": "dss_gauss",
            "enabled": False,
            "params": {"a": 1.0, "n_components": 5},
        },
        {"name": "dss_skew", "enabled": False, "params": {"n_components": 5}},
        {
            "name": "dss_smooth_tanh",
            "enabled": False,
            "params": {"alpha": 1.0, "window": 10, "n_components": 5},
        },
        {
            "name": "dss_dct",
            "enabled": False,
            "params": {"cutoff_fraction": 0.5, "n_components": 5},
        },
        {
            "name": "dss_spectrogram",
            "enabled": False,
            "params": {"threshold_percentile": 90.0, "n_components": 5},
        },
        # ── Additional epoch-stage (disabled by default) ─────────────────────
        {
            "name": "dss_quasiperiodic",
            "enabled": False,
            "params": {"peak_distance": 100, "n_components": 5},
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# DENOISER REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DenoiserSpec:
    """Registry entry for a single denoiser."""

    stage: str  # "raw" (pre-epoch) | "epoch" (post-epoch)
    label: str  # Human-readable name used in plot legends
    color: str  # Matplotlib color string
    apply: Callable  # apply(mne_obj, params: dict, sfreq: float) -> mne_obj


# ── IterativeDSS reconstruction helper ────────────────────────────────────────


def _idss_apply(mne_obj, denoiser_instance, n_components):
    """Fit IterativeDSS and return a reconstructed MNE object.

    IterativeDSS.transform() returns numpy sources only (no return_type).
    inverse_transform() maps sources back to channel space.  The per-channel
    temporal mean subtracted during centring is *not* restored; for avg-ref
    EEG this is negligible.
    """
    idss = IterativeDSS(denoiser=denoiser_instance.denoise, n_components=n_components)
    idss.fit(mne_obj)
    sources = idss.transform(mne_obj)  # (n_comp, n_times) | (n_comp, n_epochs, n_times)

    if sources.ndim == 3:
        # inverse_transform expects (n_epochs, n_comp, n_times)
        rec = idss.inverse_transform(sources.transpose(1, 0, 2))
    else:
        rec = idss.inverse_transform(sources)  # (n_channels, n_times)

    out = mne_obj.copy()
    out._data[:] = rec
    return out


# ── Individual apply functions ────────────────────────────────────────────────


def _apply_baseline(data, params, sfreq):
    return data


def _apply_zapline(raw, params, sfreq):
    line_freq = params.get("line_freq", PIPELINE_CONFIG["processing"]["line_freq"])
    zap = ZapLine(
        sfreq=sfreq,
        line_freq=line_freq,
        n_remove=params.get("n_remove", "auto"),
        threshold=params.get("threshold", 3.0),
    )
    return zap.fit_transform(raw)


def _apply_dss_bandpass(raw, params, sfreq):
    bias = BandpassBias(freq_band=params["freq_band"], sfreq=sfreq)
    dss = DSS(bias=bias, n_components=params.get("n_components", 10), return_type="raw")
    dss.fit(raw)
    return dss.transform(raw)


def _apply_dss_tsr(raw, params, sfreq):
    bias = TimeShiftBias(shifts=params.get("shifts", 20))
    dss = DSS(bias=bias, n_components=params.get("n_components", 10), return_type="raw")
    dss.fit(raw)
    return dss.transform(raw)


def _apply_dss_smoothing(raw, params, sfreq):
    bias = SmoothingBias(
        window=params.get("window", 10),
        iterations=params.get("iterations", 1),
    )
    dss = DSS(bias=bias, n_components=params.get("n_components", 10), return_type="raw")
    dss.fit(raw)
    return dss.transform(raw)


def _apply_dss_linenoise(raw, params, sfreq):
    line_freq = params.get("freq", PIPELINE_CONFIG["processing"]["line_freq"])
    bias = LineNoiseBias(freq=line_freq, sfreq=sfreq)
    dss = DSS(bias=bias, n_components=params.get("n_components", 5), return_type="raw")
    dss.fit(raw)
    return dss.transform(raw)


def _apply_dss_peak(raw, params, sfreq):
    bias = PeakFilterBias(freq=params["freq"], sfreq=sfreq)
    dss = DSS(bias=bias, n_components=params.get("n_components", 5), return_type="raw")
    dss.fit(raw)
    return dss.transform(raw)


def _apply_dss_comb(raw, params, sfreq):
    bias = CombFilterBias(
        fundamental_freq=params["fundamental_freq"],
        sfreq=sfreq,
        n_harmonics=params.get("n_harmonics", 3),
    )
    dss = DSS(bias=bias, n_components=params.get("n_components", 5), return_type="raw")
    dss.fit(raw)
    return dss.transform(raw)


def _apply_dss_average(epochs, params, sfreq):
    """DSS keeping components reproducible across trials — ERP enhancement.

    AverageBias finds spatial filters maximising cov(trial-average) / cov(all
    trials), i.e. the subspace with highest evoked SNR.  Top components are the
    most consistent across trials; discarding the rest removes non-reproducible
    (noise) components entirely rather than just attenuating them.
    """
    bias = AverageBias(axis="epochs")
    dss = DSS(
        bias=bias, n_components=params.get("n_components", 5), return_type="epochs"
    )
    dss.fit(epochs)
    return dss.transform(epochs)


def _apply_dss_kurtosis(raw, params, sfreq):
    nd = KurtosisDenoiser(
        nonlinearity=params.get("nonlinearity", "tanh"),
        alpha=params.get("alpha", 1.0),
    )
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_wiener(raw, params, sfreq):
    nd = WienerMaskDenoiser(
        window_samples=params.get("window_samples", 50),
        noise_percentile=params.get("noise_percentile", 25.0),
    )
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_tanh(raw, params, sfreq):
    nd = TanhMaskDenoiser(alpha=params.get("alpha", 1.0))
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_robust_tanh(raw, params, sfreq):
    nd = RobustTanhDenoiser(alpha=params.get("alpha", 1.0))
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_gauss(raw, params, sfreq):
    nd = GaussDenoiser(a=params.get("a", 1.0))
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_skew(raw, params, sfreq):
    nd = SkewDenoiser()
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_smooth_tanh(raw, params, sfreq):
    nd = SmoothTanhDenoiser(
        alpha=params.get("alpha", 1.0),
        window=params.get("window", 10),
    )
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_dct(raw, params, sfreq):
    nd = DCTDenoiser(cutoff_fraction=params.get("cutoff_fraction", 0.5))
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_spectrogram(raw, params, sfreq):
    nd = SpectrogramDenoiser(
        threshold_percentile=params.get("threshold_percentile", 90.0)
    )
    return _idss_apply(raw, nd, params.get("n_components", 5))


def _apply_dss_quasiperiodic(epochs, params, sfreq):
    nd = QuasiPeriodicDenoiser(
        peak_distance=params.get("peak_distance", 100),
        peak_height_percentile=params.get("peak_height_percentile", 75.0),
    )
    return _idss_apply(epochs, nd, params.get("n_components", 5))


# ── Registry ──────────────────────────────────────────────────────────────────

DENOISER_REGISTRY: dict[str, DenoiserSpec] = {
    # name; stage; label; color; apply fn
    "baseline": DenoiserSpec("raw", "Baseline (ICA only)", "black", _apply_baseline),
    "zapline": DenoiserSpec("raw", "ZapLine", "tab:blue", _apply_zapline),
    "dss_bandpass": DenoiserSpec(
        "raw", "DSS bandpass", "tab:orange", _apply_dss_bandpass
    ),
    "dss_tsr": DenoiserSpec("raw", "DSS TSR", "tab:green", _apply_dss_tsr),
    "dss_smoothing": DenoiserSpec(
        "raw", "DSS smoothing", "tab:cyan", _apply_dss_smoothing
    ),
    "dss_linenoise": DenoiserSpec(
        "raw", "DSS line noise", "tab:red", _apply_dss_linenoise
    ),
    "dss_peak": DenoiserSpec("raw", "DSS peak filter", "tab:pink", _apply_dss_peak),
    "dss_comb": DenoiserSpec("raw", "DSS comb filter", "tab:olive", _apply_dss_comb),
    "dss_kurtosis": DenoiserSpec(
        "raw", "DSS kurtosis", "tab:brown", _apply_dss_kurtosis
    ),
    "dss_wiener": DenoiserSpec("raw", "DSS Wiener mask", "tab:gray", _apply_dss_wiener),
    "dss_tanh": DenoiserSpec("raw", "DSS tanh mask", "#e377c2", _apply_dss_tanh),
    "dss_robust_tanh": DenoiserSpec(
        "raw", "DSS robust tanh", "#17becf", _apply_dss_robust_tanh
    ),
    "dss_gauss": DenoiserSpec("raw", "DSS gauss mask", "#bcbd22", _apply_dss_gauss),
    "dss_skew": DenoiserSpec("raw", "DSS skew", "#aec7e8", _apply_dss_skew),
    "dss_smooth_tanh": DenoiserSpec(
        "raw", "DSS smooth tanh", "#ffbb78", _apply_dss_smooth_tanh
    ),
    "dss_dct": DenoiserSpec("raw", "DSS DCT lowpass", "#98df8a", _apply_dss_dct),
    "dss_spectrogram": DenoiserSpec(
        "raw", "DSS spectrogram", "#ff9896", _apply_dss_spectrogram
    ),
    "dss_average": DenoiserSpec(
        "epoch", "DSS average", "tab:purple", _apply_dss_average
    ),
    "dss_quasiperiodic": DenoiserSpec(
        "epoch", "DSS quasi-periodic", "#9467bd", _apply_dss_quasiperiodic
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _get_stimulus_rename_map(descriptions):
    """Build annotation rename map from trialStart descriptions."""
    rename_map = {}
    for desc in descriptions:
        if not desc.startswith("trialStart"):
            continue
        cond_m = re.search(r"condition:(\w+)", desc)
        stim_m = re.search(r"stimulus:(\w+)", desc)
        if cond_m and stim_m:
            rename_map[desc] = f"{cond_m.group(1)}/{stim_m.group(1)}"
    return rename_map


def _pick_erp_channel(inst, channel_priority):
    """Return the first available channel from the priority list."""
    for ch in channel_priority:
        if ch in inst.ch_names:
            return ch
    return inst.ch_names[0]


def get_epochs(raw, cond, proc):
    """Filter, epoch, and split a Raw object for one condition.

    Returns
    -------
    epochs : mne.Epochs
        All trials (both labels).
    epochs_dict : dict[str, mne.Epochs]
        Trials split by congruency label.
    """
    rename_map = _get_stimulus_rename_map(raw.annotations.description)
    remap = REMAPS[cond]

    raw_remap = raw.copy()
    raw_remap.annotations.rename(rename_map)
    raw_remap.annotations.rename(remap)
    raw_remap.filter(
        l_freq=proc["bandpass_erp"][0],
        h_freq=proc["bandpass_erp"][1],
        verbose=False,
    )

    mask = [a in remap.values() for a in raw_remap.annotations.description]
    raw_remap.set_annotations(raw_remap.annotations[mask])

    events, ids = mne.events_from_annotations(raw_remap, verbose=False)
    id_of_interest = {ev: ids[ev] for ev in remap.values()}

    epochs = mne.Epochs(
        raw_remap,
        event_id=id_of_interest,
        events=events,
        tmin=proc["epoch_tmin"],
        tmax=proc["epoch_tmax"],
        baseline=proc["baseline"],
        preload=True,
        verbose=False,
    )
    epochs_dict = {label: epochs[label] for label in remap.values()}
    return epochs, epochs_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Data extraction
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_psd(raw, fmax=120.0):
    """Channel-averaged PSD for EEG channels."""
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks="eeg")
    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 4), axis=1)
    mask = freqs <= fmax
    return freqs[mask], np.mean(psd[:, mask], axis=0)


def _compute_erp_stats(epochs_dict, ch):
    """Compute per-condition ERPs, difference wave, and propagated SEM.

    SEM_diff = sqrt(SEM_incongruent² + SEM_congruent²)

    Returns
    -------
    times : ndarray (n_times,)
    diff  : ndarray (n_times,)   incongruent − congruent, µV
    sem   : ndarray (n_times,)   propagated SEM across trials, µV
    erp   : dict[str, dict]      per-condition {"mean", "sem"} in µV
    """
    stats = {}
    times = None
    for label, ep in epochs_dict.items():
        data = ep.copy().pick([ch]).get_data()[:, 0, :] * 1e6  # (n_trials, n_times)
        times = ep.times
        n = len(data)
        stats[label] = {
            "mean": data.mean(axis=0),
            "sem": data.std(axis=0) / np.sqrt(n) if n > 1 else np.zeros(data.shape[1]),
        }

    zeros = np.zeros_like(times)
    diff = (
        stats.get("incongruent", {"mean": zeros})["mean"]
        - stats.get("congruent", {"mean": zeros})["mean"]
    )
    sem = np.sqrt(
        stats.get("incongruent", {"sem": zeros})["sem"] ** 2
        + stats.get("congruent", {"sem": zeros})["sem"] ** 2
    )
    return times, diff, sem, stats


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════


def _plot_erp_line(ax, times_ms, diff, sem, color, label, alpha_fill=0.15):
    """Plot a difference wave with ± SEM/std shading."""
    ax.plot(times_ms, diff, color=color, label=label, lw=1.5, alpha=0.9)
    ax.fill_between(
        times_ms, diff - sem, diff + sem, color=color, alpha=alpha_fill, linewidth=0
    )


def plot_subject_comparison(subject_data, cond, ch, fname_out, proc):
    """Save single-subject comparison figure.

    Panel 1 : channel-averaged PSD (pre-epoch denoisers).
    Panel 2 : congruent and incongruent ERPs ± SEM (pre-subtraction).
    Panel 3 : difference wave incongruent − congruent ± SEM.

    Solid lines = incongruent, dashed lines = congruent.
    """
    fig, (ax_psd, ax_erp_raw, ax_erp_diff) = plt.subplots(1, 3, figsize=(20, 5))
    task_type = (
        "dual-task (walking)" if cond.startswith("DT") else "single-task (seated)"
    )
    fig.suptitle(
        f"Denoiser comparison — {cond}  [{task_type}]  |  channel: {ch}\n"
        "Shading = ± SEM across trials"
    )

    # Panel 1 — PSD (pre-epoch denoisers only)
    ax_psd.set_title("Channel-averaged PSD (EEG)")
    for name, d in subject_data.items():
        if d["freqs"] is None:
            continue
        ax_psd.semilogy(
            d["freqs"],
            d["psd"],
            color=d["color"],
            label=d["label"],
            lw=1.5,
            alpha=0.85,
        )
    ax_psd.axvline(
        proc["line_freq"],
        color="red",
        ls="--",
        alpha=0.4,
        label=f"{proc['line_freq']:.0f} Hz",
    )
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (V²/Hz)")
    ax_psd.set_xlim(0, 120)
    ax_psd.legend(fontsize=8)

    # Panel 2 — ERPs pre-subtraction (solid = incongruent, dashed = congruent)
    ax_erp_raw.set_title(f"ERPs per condition  ({ch})")
    for name, d in subject_data.items():
        if d["times"] is None or not d["erp"]:
            continue
        times_ms = d["times"] * 1000
        color = d["color"]
        for condition, ls in [("incongruent", "-"), ("congruent", "--")]:
            cstats = d["erp"].get(condition)
            if cstats is None:
                continue
            lbl = (
                f"{d['label']}  {condition}"
                if condition == "incongruent"
                else "_nolegend_"
            )
            ax_erp_raw.plot(
                times_ms,
                cstats["mean"],
                color=color,
                ls=ls,
                lw=1.5,
                alpha=0.9,
                label=lbl,
            )
            ax_erp_raw.fill_between(
                times_ms,
                cstats["mean"] - cstats["sem"],
                cstats["mean"] + cstats["sem"],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax_erp_raw.axhline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_raw.axvline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_raw.set_xlabel("Time (ms)")
    ax_erp_raw.set_ylabel("Amplitude (µV)")
    ax_erp_raw.legend(fontsize=7, title="— incong  ╌ cong", title_fontsize=7)

    # Panel 3 — difference wave
    ax_erp_diff.set_title(f"Difference wave: incongruent − congruent  ({ch})")
    for name, d in subject_data.items():
        if d["times"] is None:
            continue
        _plot_erp_line(
            ax_erp_diff,
            d["times"] * 1000,
            d["diff"],
            d["sem"],
            color=d["color"],
            label=d["label"],
        )
    ax_erp_diff.axhline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_diff.axvline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_diff.set_xlabel("Time (ms)")
    ax_erp_diff.set_ylabel("Amplitude (µV)")
    ax_erp_diff.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname_out, dpi=200)
    print(f"  Saved: {fname_out.name}")
    plt.close(fig)
    gc.collect()


def plot_group_comparison(all_subject_data, cond, fname_out, proc):
    """Save group-level comparison figure.

    Panel 1 : PSD group mean ± std in log10 space.
    Panel 2 : group mean ERPs per condition ± std across subjects.
    Panel 3 : group mean difference wave ± std across subjects.

    Solid lines = incongruent, dashed lines = congruent.
    """
    if not all_subject_data:
        return

    n_subj = len(all_subject_data)
    fig, (ax_psd, ax_erp_raw, ax_erp_diff) = plt.subplots(1, 3, figsize=(20, 5))
    task_type = (
        "dual-task (walking)" if cond.startswith("DT") else "single-task (seated)"
    )
    fig.suptitle(
        f"Group denoiser comparison — {cond}  [{task_type}]  (N={n_subj})\n"
        "Shading = ± std across subjects"
    )

    all_names = list(dict.fromkeys(name for s in all_subject_data for name in s))

    # Panel 1 — PSD group mean ± std (log10 space, pre-epoch only)
    ax_psd.set_title("Channel-averaged PSD (EEG) — group")
    for name in all_names:
        entries = [
            s[name]
            for s in all_subject_data
            if name in s and s[name]["freqs"] is not None
        ]
        if not entries:
            continue
        # Build common frequency grid from the entry with the most points
        freqs = max((e["freqs"] for e in entries), key=len)
        color = entries[0]["color"]
        label = entries[0]["label"]
        interp_psds = [np.interp(freqs, e["freqs"], e["psd"]) for e in entries]
        log_psds = np.log10(np.array(interp_psds))
        mean_log = log_psds.mean(axis=0)
        std_log = log_psds.std(axis=0)
        ax_psd.semilogy(freqs, 10**mean_log, color=color, label=label, lw=1.5)
        ax_psd.fill_between(
            freqs,
            10 ** (mean_log - std_log),
            10 ** (mean_log + std_log),
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax_psd.axvline(
        proc["line_freq"],
        color="red",
        ls="--",
        alpha=0.4,
        label=f"{proc['line_freq']:.0f} Hz",
    )
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (V²/Hz)")
    ax_psd.set_xlim(0, 120)
    ax_psd.legend(fontsize=8)

    # Panel 2 — group mean ERPs per condition ± std across subjects
    ax_erp_raw.set_title("ERPs per condition — group")
    for name in all_names:
        entries = [s[name] for s in all_subject_data if name in s and s[name]["erp"]]
        if not entries:
            continue
        times = entries[0]["times"]
        color = entries[0]["color"]
        label = entries[0]["label"]
        for condition, ls in [("incongruent", "-"), ("congruent", "--")]:
            cond_means = [
                e["erp"][condition]["mean"] for e in entries if condition in e["erp"]
            ]
            if not cond_means:
                continue
            arr = np.array(cond_means)
            lbl = (
                f"{label}  {condition}" if condition == "incongruent" else "_nolegend_"
            )
            ax_erp_raw.plot(
                times * 1000,
                arr.mean(axis=0),
                color=color,
                ls=ls,
                lw=1.5,
                alpha=0.9,
                label=lbl,
            )
            ax_erp_raw.fill_between(
                times * 1000,
                arr.mean(axis=0) - arr.std(axis=0),
                arr.mean(axis=0) + arr.std(axis=0),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax_erp_raw.axhline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_raw.axvline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_raw.set_xlabel("Time (ms)")
    ax_erp_raw.set_ylabel("Amplitude (µV)")
    ax_erp_raw.legend(fontsize=7, title="— incong  ╌ cong", title_fontsize=7)

    # Panel 3 — group mean difference wave ± std across subjects
    ax_erp_diff.set_title("Difference wave: incongruent − congruent  (group)")
    for name in all_names:
        entries = [
            s[name]
            for s in all_subject_data
            if name in s and s[name]["diff"] is not None
        ]
        if not entries:
            continue
        times = entries[0]["times"]
        color = entries[0]["color"]
        label = entries[0]["label"]
        diffs = np.array([e["diff"] for e in entries])
        _plot_erp_line(
            ax_erp_diff,
            times * 1000,
            diffs.mean(axis=0),
            diffs.std(axis=0),
            color=color,
            label=f"{label}  (n={len(entries)})",
        )
    ax_erp_diff.axhline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_diff.axvline(0, color="gray", ls="--", alpha=0.4)
    ax_erp_diff.set_xlabel("Time (ms)")
    ax_erp_diff.set_ylabel("Amplitude (µV)")
    ax_erp_diff.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname_out, dpi=200)
    print(f"  Saved (group): {fname_out.name}")
    plt.close(fig)
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_comparison(fname_clean, cond, overwrite=False):
    """Run all enabled denoisers on one preprocessed file.

    Reads PIPELINE_CONFIG to determine which denoisers to apply and at what
    stage.  Pre-epoch (raw-stage) denoisers are applied before epoching; each
    produces an independent denoised Raw that is then epoched separately.
    Post-epoch (epoch-stage) denoisers are applied to the epoched *baseline*
    (no extra pre-epoch denoising) output.

    Returns
    -------
    subject_data : dict
        {denoiser_name: {"times", "diff", "sem", "freqs", "psd",
                         "label", "color", "stage"}}
        freqs/psd are None for epoch-stage denoisers.
    ch : str
        ERP channel selected for this subject.
    """
    config = PIPELINE_CONFIG
    proc = config["processing"]

    stem = fname_clean.name.replace(".fif.gz", "")
    out_dir = fname_clean.parent / "denoise_comparison"
    out_dir.mkdir(exist_ok=True)
    fname_plot = out_dir / f"{stem}_comparison.png"

    if fname_plot.exists() and not overwrite:
        print(f"  Skipping (exists): {fname_plot.name}")

    print(f"\n--- {fname_clean.name}  [{cond}] ---")
    raw_clean = mne.io.read_raw_fif(fname_clean, preload=True, verbose=False)
    print(
        f"  {raw_clean.info['nchan']} channels, "
        f"{raw_clean.info['sfreq']:.0f} Hz, "
        f"{raw_clean.times[-1]:.1f} s"
    )

    # Prehook
    if config["prehook"] is not None:
        raw_clean = config["prehook"](raw_clean)

    sfreq = raw_clean.info["sfreq"]
    ch = _pick_erp_channel(raw_clean, proc["erp_channels"])

    # Resolve enabled denoisers — attach final label/color from registry + overrides
    enabled: list[dict] = []
    for d_cfg in config["denoisers"]:
        if not d_cfg["enabled"]:
            continue
        name = d_cfg["name"]
        if name not in DENOISER_REGISTRY:
            print(f"  WARNING: unknown denoiser '{name}' — skipping")
            continue
        spec = DENOISER_REGISTRY[name]
        enabled.append(
            {
                "name": name,
                "params": d_cfg.get("params", {}),
                "label": d_cfg.get("label", spec.label),
                "color": d_cfg.get("color", spec.color),
                "stage": spec.stage,
                "apply": spec.apply,
            }
        )

    # ── Pre-epoch (raw-stage) denoisers ──────────────────────────────────────
    raws: dict[str, mne.io.BaseRaw] = {}
    for d in enabled:
        if d["stage"] != "raw":
            continue
        name = d["name"]
        if name == "baseline":
            raws[name] = raw_clean
        else:
            print(f"  Applying raw-stage denoiser: {d['label']}...")
            raws[name] = d["apply"](raw_clean, d["params"], sfreq)

    # ── Epoch all pre-epoch versions ──────────────────────────────────────────
    print("  Epoching pre-epoch denoiser outputs...")
    epochs_raw: dict[str, dict] = {}
    baseline_epochs = None
    for name, raw in raws.items():
        full_epochs, ep_dict = get_epochs(raw, cond, proc)
        epochs_raw[name] = ep_dict
        if name == "baseline":
            baseline_epochs = full_epochs

    # ── Post-epoch (epoch-stage) denoisers ───────────────────────────────────
    epoch_denoisers = [d for d in enabled if d["stage"] == "epoch"]
    epochs_epoch: dict[str, dict] = {}
    if epoch_denoisers and baseline_epochs is None:
        # No raw-stage baseline in enabled list; epoch raw_clean directly
        baseline_epochs, _ = get_epochs(raw_clean, cond, proc)

    for d in epoch_denoisers:
        name = d["name"]
        print(f"  Applying epoch-stage denoiser: {d['label']}...")
        denoised = d["apply"](baseline_epochs, d["params"], sfreq)
        epochs_epoch[name] = {lbl: denoised[lbl] for lbl in REMAPS[cond].values()}

    # ── Build subject_data ────────────────────────────────────────────────────
    all_epoch_dicts = {**epochs_raw, **epochs_epoch}
    subject_data: dict[str, dict] = {}
    meta = {d["name"]: d for d in enabled}

    for name, ep_dict in all_epoch_dicts.items():
        times, diff, sem, erp = _compute_erp_stats(ep_dict, ch)
        m = meta[name]
        subject_data[name] = {
            "times": times,
            "diff": diff,
            "sem": sem,
            "erp": erp,  # {"incongruent": {"mean", "sem"}, "congruent": {...}}
            "freqs": None,
            "psd": None,
            "label": m["label"],
            "color": m["color"],
            "stage": m["stage"],
        }

    for name, raw in raws.items():
        if name in subject_data:
            freqs, psd = _compute_psd(raw)
            subject_data[name]["freqs"] = freqs
            subject_data[name]["psd"] = psd

    # ── Single-subject plot ───────────────────────────────────────────────────
    if not config.get("group_only", False):
        print("  Plotting (single subject)...")
        plot_subject_comparison(subject_data, cond, ch, fname_plot, proc)
    else:
        print("  Skipping single-subject plot (group_only=True).")

    return subject_data, ch


def main():
    """Find all preprocessed clean files, run per-subject and group comparisons."""
    fnames = sorted(DERIV_DIR.rglob("*_preproc_asr.fif.gz"))
    if not fnames:
        print(f"No *_preproc_asr.fif.gz files found under {DERIV_DIR}")
        return

    print(f"Found {len(fnames)} file(s) to process.")

    group_data: dict[str, list] = {}

    exclude = set(PIPELINE_CONFIG.get("exclude_subjects", []))

    for fname in fnames:
        subject = fname.parent.name
        if subject in exclude:
            print(f"  Skipping (excluded): {subject}")
            continue
        cond = next((c for c in REMAPS if c in fname.name), None)
        if cond is None:
            print(f"  Skipping (no condition match): {fname.name}")
            continue
        try:
            subject_data, _ = run_comparison(fname, cond, overwrite=True)
            group_data.setdefault(cond, []).append(subject_data)
        except Exception as e:
            print(f"  ERROR on {fname.name}: {e}")
            raise  # remove to continue on errors

    # ── Group plots (one per condition) ───────────────────────────────────────
    proc = PIPELINE_CONFIG["processing"]
    print("\n--- Group comparisons ---")
    for cond, subjects in group_data.items():
        if len(subjects) < 2:
            print(f"  Skipping group plot for {cond} (only {len(subjects)} subject)")
            continue
        fname_group = DERIV_DIR / f"group_{cond}_comparison.png"
        print(f"  {cond}: N={len(subjects)}")
        plot_group_comparison(subjects, cond, fname_group, proc)


# %% Entry point

if __name__ == "__main__":
    main()

# %%
