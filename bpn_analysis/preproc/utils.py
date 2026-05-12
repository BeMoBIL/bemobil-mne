"""Utility functions for preprocessing."""

# %% Imports

import json
import logging

import mne
import mne_faster
import mne_icalabel
import numpy as np
from meegkit.asr import ASR

LOGGER = logging.getLogger(__name__)

# %% Functions


def _annotate_break_iter(raw, annotate_break_kwargs):
    """Annotate break iteratively.

    Annotate break with kwargs. If breaks are too excessive, redefine what
    constitutes a break, so that less breaks are detected.
    """
    # Get block breaks and irrelevant data segments (beginning, end) as annotations
    annotate_break_kwargs = (
        dict(min_break_duration=5, t_start_after_previous=0, t_stop_before_next=0)
        if annotate_break_kwargs is None
        else dict(annotate_break_kwargs)
    )

    recording_dur = float(raw.times[-1] - raw.times[0])
    if recording_dur <= 0:
        raise RuntimeError(
            f"Found unlikely recording duration: {recording_dur} seconds"
        )

    thresh = 0.6
    max_iter = 30
    iter_count = 0

    while True:
        annots_break = mne.preprocessing.annotate_break(raw, **annotate_break_kwargs)

        total_bad_time = np.finfo(float).eps
        for desc, dur in zip(annots_break.description, annots_break.duration):
            if str(desc).lower().startswith("bad_break"):
                total_bad_time += float(dur)

        if total_bad_time < 0:
            raise RuntimeError(
                f"Negative total break duration: {total_bad_time} seconds"
            )

        prop_break = total_bad_time / recording_dur

        if prop_break >= 1.0:
            raise RuntimeError(
                f"Unlikely break/recording ratio: {prop_break:.2f} "
                f"({total_bad_time:.2f}s breaks vs {recording_dur:.2f}s total)"
            )

        if prop_break <= thresh:
            break

        iter_count += 1
        if iter_count >= max_iter:
            LOGGER.error(
                f"Stopping after {max_iter} iterations; breaks still span "
                f"{prop_break:.2f} of recording. "
                f"Last parameters: {json.dumps(annotate_break_kwargs, indent=4)}"
            )
            break

        LOGGER.warning(
            f"Breaks span a proportion of {prop_break:.2f} "
            f"of the recording (<={thresh} accepted)."
        )
        LOGGER.warning(
            "Adjusting annotate_break_kwargs from: "
            f"{json.dumps(annotate_break_kwargs, indent=4)}"
        )

        annotate_break_kwargs["min_break_duration"] += 2
        annotate_break_kwargs["t_start_after_previous"] += 0.5
        annotate_break_kwargs["t_stop_before_next"] += 0.5

        LOGGER.warning(
            f"After adjustment: {json.dumps(annotate_break_kwargs, indent=4)}"
        )

    return annots_break


def compute_ica(
    raw,
    filter_bands_ica=(1.0, 100.0),
    notch_freqs=(50, 100, 150),
    downsample_ica=250,
    thresh=0.7,
    rng_seed=None,
    exclude_labels=None,
    include_labels=None,
):
    """Fit ICA on a filtered copy of *raw* and label components with ICLabel.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording (must contain EEG channels).
    filter_bands_ica : tuple of float
        ``(l_freq, h_freq)`` for the ICA-specific bandpass filter.
    notch_freqs : array-like
        Line-noise frequencies to notch out before ICA.
    downsample_ica : float
        Target sampling rate for ICA fitting (anti-aliasing applied
        automatically).  Skipped when the recording is already at or below this
        rate.
    thresh : float
        ICLabel probability threshold.  A component is excluded only when its
        predicted probability for the artifact label meets or exceeds this value.
    rng_seed : int | None
        Random seed passed to :class:`mne.preprocessing.ICA` for
        reproducibility.
    exclude_labels : list of str | None
        ICLabel category names to *exclude* (e.g. ``["eye", "muscle"]``).
        Mutually exclusive with *include_labels*.
    include_labels : list of str | None
        ICLabel category names to *keep*; all other categories are excluded
        (e.g. ``["brain", "other"]``).
        Mutually exclusive with *exclude_labels*.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object with ``ica.exclude`` populated according to the
        label criteria.
    ic_labels : dict
        Output of :func:`mne_icalabel.label_components` containing
        ``"labels"`` and ``"y_pred_proba"`` keys.

    Raises
    ------
    ValueError
        If both *exclude_labels* and *include_labels* are provided.
    """
    if exclude_labels is not None and include_labels is not None:
        raise ValueError("Specify either exclude_labels or include_labels, not both.")

    raw_ica = raw.copy().pick("eeg")

    raw_ica.filter(l_freq=filter_bands_ica[0], h_freq=None)
    raw_ica.filter(l_freq=None, h_freq=filter_bands_ica[1])
    raw_ica.notch_filter(freqs=notch_freqs, notch_widths=1.0)

    if raw_ica.info["sfreq"] > downsample_ica:
        raw_ica.resample(downsample_ica)

    raw_ica.set_eeg_reference(ref_channels="average")

    epochs = mne.make_fixed_length_epochs(
        raw_ica, duration=1.0, preload=True, reject_by_annotation=True
    )

    bad_epochs = mne_faster.find_bad_epochs(epochs)
    if len(bad_epochs) > 0:
        epochs.drop(bad_epochs)

    ica = mne.preprocessing.ICA(
        n_components=None,
        random_state=rng_seed,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
    )
    ica.fit(epochs)

    ic_labels = mne_icalabel.label_components(epochs, ica, method="iclabel")

    if exclude_labels is not None:
        exclude_idx = [
            idx
            for idx, (label, prob) in enumerate(
                zip(ic_labels["labels"], ic_labels["y_pred_proba"])
            )
            if label in exclude_labels and prob >= thresh
        ]
    elif include_labels is not None:
        exclude_idx = [
            idx
            for idx, (label, prob) in enumerate(
                zip(ic_labels["labels"], ic_labels["y_pred_proba"])
            )
            if label not in include_labels or prob < thresh
        ]
    else:
        exclude_idx = []

    ica.exclude = exclude_idx

    return ica, ic_labels


def fit_dipoles_on_ica(ica, info, trans="fsaverage"):
    """Fit a single dipole to each ICA component topography.

    Adapted from standard_scripts. Uses the fsaverage BEM and template
    head→MRI transform by default, so no individual MRI is required.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    info : mne.Info
        Channel info the ICA was fitted on (EEG channels only).
    trans : str | Transform
        Head→MRI transform. ``"fsaverage"`` uses MNE's built-in template.

    Returns
    -------
    dipoles : list[mne.Dipole]
        One dipole per component.
    residuals : list[mne.Evoked]
        Residual field per component.
    """
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    bem = str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif")

    cov = mne.cov.make_ad_hoc_cov(info, std=dict(eeg=1))
    components = ica.get_components()  # (n_channels, n_components)

    sel = mne.channel_indices_by_type(info, picks="eeg", exclude="bads")["eeg"]
    info_clean = mne.pick_info(info, sel=sel)

    evoked = mne.EvokedArray(components, info_clean)
    evoked.set_eeg_reference(ref_channels="average")
    if not evoked.info["dev_head_t"]:
        evoked.info["dev_head_t"] = mne.Transform(fro="meg", to="head")

    dipoles, residuals = mne.fit_dipole(evoked, cov=cov, bem=bem, trans=trans, verbose=False)

    dipoles._set_times(np.zeros_like(dipoles.times))
    dipoles = [dip for dip in dipoles]
    _dat = residuals.get_data()
    residuals = [mne.EvokedArray(_dat[:, i : i + 1], info_clean) for i in range(_dat.shape[1])]

    return dipoles, residuals


def compute_dipolarity(components, info, rv_thresh=0.15, trans="fsaverage"):
    """Fraction of ICA topographies with dipole residual variance below *rv_thresh*.

    RV = 1 − GOF/100. A component is dipolar when RV < *rv_thresh* (default 15%).

    Parameters
    ----------
    components : ndarray, shape (n_channels, n_components)
        ICA mixing-matrix columns (topographies). For a standard ICA use
        ``ica.get_components()``. For a combined head+neck ICA, pass only
        the head-channel rows.
    info : mne.Info
        Channel info matching the rows of *components* (EEG channels only).
    rv_thresh : float
        Residual variance threshold in [0, 1].
    trans : str | Transform
        Passed to ``mne.fit_dipole``.

    Returns
    -------
    dict
        ``fraction_dipolar``, ``n_dipolar``, ``rv_values`` (one per component).
    """
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    bem = str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif")

    sel = mne.channel_indices_by_type(info, picks="eeg", exclude="bads")["eeg"]
    info_clean = mne.pick_info(info, sel=sel)
    components_clean = components[sel, :]

    cov = mne.cov.make_ad_hoc_cov(info_clean, std=dict(eeg=1))
    evoked = mne.EvokedArray(components_clean, info_clean)
    evoked.set_eeg_reference(ref_channels="average")
    if not evoked.info["dev_head_t"]:
        evoked.info["dev_head_t"] = mne.Transform(fro="meg", to="head")

    try:
        dipoles, _ = mne.fit_dipole(evoked, cov=cov, bem=bem, trans=trans, verbose=False)
        dipoles._set_times(np.zeros_like(dipoles.times))
        rvs = [float(1.0 - dip.gof[0] / 100.0) for dip in dipoles]
    except Exception as exc:
        LOGGER.warning(f"Dipole fitting failed: {exc}")
        return {"fraction_dipolar": np.nan, "n_dipolar": np.nan, "rv_values": []}

    n_dipolar = int(sum(rv < rv_thresh for rv in rvs))
    return {
        "fraction_dipolar": n_dipolar / len(rvs) if rvs else np.nan,
        "n_dipolar": n_dipolar,
        "rv_values": rvs,
    }


def compute_mi_reduction(raw_before, raw_after, picks="eeg"):
    """Total pairwise MI reduction under the Gaussian approximation.

    MI_total = −½ · logdet(R), where R is the channel correlation matrix.
    Independent channels → R = I → MI = 0. A good denoiser reduces MI by
    removing shared artifact variance.

    Parameters
    ----------
    raw_before, raw_after : mne.io.Raw
        Recordings before and after denoising.
    picks : str
        Channel type to include (default ``"eeg"``).

    Returns
    -------
    dict
        ``mi_before``, ``mi_after``, ``mi_reduction`` (before − after),
        ``mi_reduction_pct``.
    """
    def _gaussian_mi(data):
        R = np.corrcoef(data)
        p = R.shape[0]
        R_reg = R + 1e-6 * np.eye(p)
        sign, logdet = np.linalg.slogdet(R_reg)
        return float(-0.5 * logdet) if sign > 0 else np.nan

    mi_b = _gaussian_mi(raw_before.get_data(picks=picks))
    mi_a = _gaussian_mi(raw_after.get_data(picks=picks))
    reduction = mi_b - mi_a
    pct = (reduction / abs(mi_b) * 100.0) if (not np.isnan(mi_b) and mi_b != 0) else np.nan
    return {
        "mi_before": mi_b,
        "mi_after": mi_a,
        "mi_reduction": reduction,
        "mi_reduction_pct": pct,
    }


def compute_asr(raw, cutoff=20, estimator="scm"):
    """Apply ASR to the EEG channels of *raw*.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.  Non-EEG channels are left untouched.
    cutoff : float
        ASR cutoff parameter (standard deviations above the clean baseline
        before a component is reconstructed).  Lower = more aggressive.
        Typical range 5–20; 20 is conservative.
    estimator : str
        Covariance estimator passed to :class:`meegkit.asr.ASR`.
        ``"scm"`` is the meegkit default (sample covariance matrix).
        ``"lwf"`` (Ledoit-Wolf) is more robust when channel count is high
        relative to the calibration window length.

    Returns
    -------
    raw_asr : mne.io.Raw
        Copy of *raw* with ASR applied to EEG channels.
    """
    eeg_idx = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_data = raw.get_data(picks=eeg_idx)

    asr = ASR(sfreq=raw.info["sfreq"], cutoff=cutoff, estimator=estimator)
    asr.fit(eeg_data)
    eeg_clean = np.real(asr.transform(eeg_data))

    raw_asr = raw.copy()
    raw_asr._data[eeg_idx] = eeg_clean

    return raw_asr
