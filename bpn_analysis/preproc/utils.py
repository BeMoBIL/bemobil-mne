"""Utility functions for preprocessing."""

# %% Imports

import datetime
import inspect
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mne
import mne_faster
import mne_icalabel
import numpy as np
from meegkit.asr import ASR

LOGGER = logging.getLogger(__name__)

# %% Constants

# Subset of channels to return when calling get_raw_subset() without a specific list
SUBSET_CHANNELS = [
    "Cz",
    "C1",
    "C2",
    "F1",
    "F2",
    "F3",
    "F4",
    "Fz",
]

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

    return annots_break, annotate_break_kwargs


# %% Provenance / descriptor helpers


def init_descriptor(source=None, pipeline=""):
    """Initialise a provenance descriptor dict for a processing run.

    Parameters
    ----------
    source : str | Path | list | None
        Input file name(s) or identifier for the source data.
    pipeline : str
        Name of the pipeline that initialised this descriptor.

    Returns
    -------
    dict
        Descriptor with keys ``pipeline``, ``input``, ``timestamp``,
        ``versions``, and an empty ``steps`` list.
    """
    try:
        bpn_ver = version("bpn-analysis")
    except PackageNotFoundError:
        bpn_ver = "unknown"

    try:
        mne_ver = version("mne")
    except PackageNotFoundError:
        mne_ver = "unknown"

    if isinstance(source, list):
        src = [str(s) for s in source]
    elif source is not None:
        src = str(source)
    else:
        src = None

    return {
        "pipeline": pipeline,
        "input": src,
        "timestamp": datetime.datetime.now().isoformat(),
        "versions": {"bpn_analysis": bpn_ver, "mne": mne_ver},
        "steps": [],
    }


def get_descriptor(raw):
    """Return the provenance descriptor stored in *raw*, or ``None``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose ``info['description']`` may hold a JSON descriptor.

    Returns
    -------
    dict | None
        Parsed descriptor, or ``None`` if not present / not valid JSON.
    """
    desc_str = raw.info.get("description", None)
    if not desc_str:
        return None
    try:
        return json.loads(desc_str)
    except (json.JSONDecodeError, TypeError):
        return None


def set_descriptor(raw, descriptor):
    """Serialise *descriptor* and store it in ``raw.info['description']``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object to annotate.
    descriptor : dict
        Provenance descriptor as returned by :func:`init_descriptor`.
    """
    raw.info["description"] = json.dumps(descriptor)


def append_desc(raw, name, **kwargs):
    """Append a named processing step to the provenance descriptor.

    If no descriptor has been initialised yet, a new one is created
    automatically (with ``pipeline="unknown"``).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose descriptor to update in place.
    name : str
        Name of the processing step.
    **kwargs
        Arbitrary key/value metadata to record for this step
        (e.g. filter frequencies, random seed, version strings).
    """
    desc = get_descriptor(raw)
    if desc is None:
        desc = init_descriptor(pipeline="unknown")
    desc["steps"].append({"name": name, **kwargs})
    set_descriptor(raw, desc)


def sig_params(func, **kwargs):
    """Return only the kwargs that match *func*'s signature.

    Useful for recording exactly which parameters were forwarded to an MNE
    function in provenance metadata without capturing irrelevant extras.

    Parameters
    ----------
    func : callable
        The function whose signature to filter against.
    **kwargs
        All keyword arguments that were (or will be) passed to *func*.

    Returns
    -------
    dict
        Subset of *kwargs* whose keys appear in *func*'s parameter list.
    """
    try:
        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (ValueError, TypeError):
        return kwargs


# %% Timing helpers


def format_duration(seconds):
    """Format a duration in seconds as a compact human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        E.g. ``"42.3s"``, ``"3m 05.0s"``, or ``"1h 12m 30.0s"``.
    """
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {int(minutes):02d}m {secs:04.1f}s"
    return f"{int(minutes)}m {secs:04.1f}s"


class StepTimer:
    """Record and log wall-clock durations of named pipeline steps.

    Parameters
    ----------
    logger : logging.Logger | None
        Logger used to emit per-step messages.  Defaults to the module logger.

    Attributes
    ----------
    timings : list of dict
        ``{"name": str, "duration_s": float}`` entries in insertion order.
    """

    def __init__(self, logger=None):
        self._logger = logger or LOGGER
        self.timings = []

    def log_step(self, name, duration_s):
        """Record and log a single step duration.

        Parameters
        ----------
        name : str
            Name of the step.
        duration_s : float
            Duration of the step in seconds.
        """
        self.timings.append({"name": name, "duration_s": float(duration_s)})
        self._logger.info(f"Step '{name}' took {format_duration(duration_s)}")

    @property
    def total_s(self):
        """float: Total recorded duration across all steps."""
        return sum(t["duration_s"] for t in self.timings)

    def format_summary(self):
        """Return a multi-line, human-readable summary of all step timings."""
        total = self.total_s
        lines = ["Preprocessing step timings:"]
        for t in self.timings:
            share = (t["duration_s"] / total * 100) if total else 0.0
            lines.append(
                f"  {t['name']:<24} {format_duration(t['duration_s']):>12}"
                f"  ({share:4.1f}%)"
            )
        lines.append(f"  {'TOTAL':<24} {format_duration(total):>12}")
        return "\n".join(lines)


# %% System info


def build_sys_info(source_data=None):
    """Return a string with full environment and version information.

    Parameters
    ----------
    source_data : list of Path | None
        Optional list of input file paths to include in the report.

    Returns
    -------
    str
        Multi-section plain-text report.
    """
    sections = []

    if source_data:
        lines = ["Source data", "-----------", ""]
        lines.extend(str(Path(p).resolve()) for p in source_data)
        sections.append("\n".join(lines))

    pkg_lines = ["Installed packages", "------------------", ""]
    for pkg in (
        "bpn-analysis",
        "mne",
        "mne-icalabel",
        "mne-faster",
        "meegkit",
        "pyprep",
    ):
        try:
            ver = version(pkg)
        except PackageNotFoundError:
            ver = "not installed"
        pkg_lines.append(f"{pkg:<20} {ver}")
    sections.append("\n".join(pkg_lines))

    buf = io.StringIO()
    mne.sys_info(fid=buf)
    sections.append("MNE SYS_INFO\n" + "-" * 20 + "\n" + buf.getvalue())

    pip_out = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--editable"],
        capture_output=True,
        text=True,
    ).stdout
    sections.append("Editable installs\n" + "-" * 17 + "\n\n" + pip_out)

    _env_conda = os.environ.get("CONDA_EXE")
    conda_exe = shutil.which("conda") or (
        shutil.which(_env_conda) if _env_conda else None
    )
    if conda_exe:
        conda_result = subprocess.run(
            [conda_exe, "env", "export"], capture_output=True, text=True
        )
        if conda_result.returncode == 0:
            sections.append("conda export\n" + "-" * 12 + "\n" + conda_result.stdout)

    return "\n\n".join(sections)


# %% Channel subset helper


def get_raw_subset(raw, subset_chs=None):
    """Return a copy of *raw* containing only the requested subset of channels.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Source recording.
    subset_chs : list of str | None
        Channel names to keep.  Defaults to :data:`SUBSET_CHANNELS`
        (frontal channels of the equidistant cap).

    Returns
    -------
    mne.io.Raw | None
        Copy of *raw* with only the available subset channels, or ``None``
        if none of the requested channels are present.
    """
    if subset_chs is None:
        subset_chs = SUBSET_CHANNELS

    subset_chs = list(subset_chs)
    present = [ch for ch in subset_chs if ch in raw.ch_names]
    missing = [ch for ch in subset_chs if ch not in raw.ch_names]

    if not present:
        LOGGER.warning("No requested subset channels available in raw. Returning None.")
        return None

    if missing:
        LOGGER.warning(
            f"Requested subset channels not in raw: {missing}. "
            f"Using available subset: {present}"
        )

    raw_subset = raw.copy().pick(present)
    LOGGER.info(f"Created subset raw with channels: {raw_subset.ch_names}")
    return raw_subset


# %% Coregistration / transform helpers


def auto_coreg_fsaverage(info, subjects_dir, fit_icp_kwargs=None):
    """Automatically coregister head to fsaverage MRI and return the transform.

    Parameters
    ----------
    info : mne.Info
        Info object from the raw recording (must have digitisation points).
    subjects_dir : str | Path
        FreeSurfer subjects directory (parent of the ``fsaverage`` folder).
    fit_icp_kwargs : dict | None
        Keyword arguments forwarded to :meth:`mne.coreg.Coregistration.fit_icp`.
        If ``None``, sensible defaults are used.

    Returns
    -------
    trans : mne.transforms.Transform
        Head-to-MRI transformation.
    """
    fiducials = "estimated"
    subject = "fsaverage"
    coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials=fiducials)
    coreg.set_scale_mode("uniform")
    coreg.fit_fiducials(verbose=True)

    if fit_icp_kwargs is None:
        fit_icp_kwargs = dict(
            n_iterations=40,
            lpa_weight=1.0,
            nasion_weight=1.0,
            rpa_weight=1.0,
            hsp_weight=0,
            eeg_weight=5.0,
            hpi_weight=0,
            verbose=True,
        )
    coreg.fit_icp(**fit_icp_kwargs)
    return coreg.trans


def _handle_trans(trans, info=None):
    """Resolve the head-to-MRI transform for dipole fitting.

    Parameters
    ----------
    trans : mne.transforms.Transform | ``"fit"`` | ``"fsaverage"`` | None
        - ``None`` or ``"fsaverage"``: use the MNE built-in fsaverage transform.
        - ``"fit"``: run automatic coregistration via
          :func:`auto_coreg_fsaverage`.
        - :class:`mne.transforms.Transform`: used directly.
    info : mne.Info | None
        Required only when ``trans="fit"``.

    Returns
    -------
    mne.transforms.Transform
    """
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    fpath_fsaverage_trans = fs_dir / "bem" / "fsaverage-trans.fif"

    if trans is None or (isinstance(trans, str) and trans == "fsaverage"):
        return mne.transforms.read_trans(fpath_fsaverage_trans)
    elif isinstance(trans, str) and trans == "fit":
        return auto_coreg_fsaverage(info, fs_dir.parent)
    else:
        if not isinstance(trans, mne.transforms.Transform):
            raise TypeError(f"Invalid trans: {trans!r}")
        return trans


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

    dipoles, residuals = mne.fit_dipole(
        evoked, cov=cov, bem=bem, trans=trans, verbose=False
    )

    dipoles._set_times(np.zeros_like(dipoles.times))
    dipoles = [dip for dip in dipoles]
    _dat = residuals.get_data()
    residuals = [
        mne.EvokedArray(_dat[:, i : i + 1], info_clean) for i in range(_dat.shape[1])
    ]

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
        dipoles, _ = mne.fit_dipole(
            evoked, cov=cov, bem=bem, trans=trans, verbose=False
        )
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
    pct = (
        (reduction / abs(mi_b) * 100.0)
        if (not np.isnan(mi_b) and mi_b != 0)
        else np.nan
    )
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
