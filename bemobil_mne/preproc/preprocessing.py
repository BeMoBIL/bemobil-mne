"""EEG preprocessing pipeline."""

# %% Imports

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import mne
import mne_faster
import numpy as np
from pyprep import NoisyChannels

from bemobil_mne.io.utils import NumpyEncoder as _NumpyEncoder
from bemobil_mne.preproc.utils import (
    StepTimer,
    _annotate_break_iter,
    _handle_trans,
    append_desc,
    compute_asr,
    compute_ica,
    detect_bad_by_line_noise,
    fit_dipoles_on_ica,
    init_descriptor,
    remove_line,
    set_descriptor,
    sig_params,
)

# Keep NumpyEncoder importable from this module for backward compatibility
NumpyEncoder = _NumpyEncoder

LOGGER = logging.getLogger(__name__)

# All ICLabel classes except "eye blink" (BeMoBIL: iclabel_classes = [1 2 4 5 6 7])
_DEFAULT_INCLUDE_LABELS = frozenset(
    {
        "brain",
        "muscle artifact",
        "heart beat",
        "line noise",
        "channel noise",
        "other",
    }
)

# %% Functions


def _template_montage_1005():
    """Return the 10-05 template montage for the installed MNE version.

    MNE 1.13 renamed ``standard_1005`` to ``colin27_1005`` (the old name is
    removed in MNE 1.14).
    """
    if "colin27_1005" in mne.channels.get_builtin_montages():
        return mne.channels.make_standard_montage("colin27_1005")
    return mne.channels.make_standard_montage("standard_1005")


def _output_stem(fname_out):
    """Return *fname_out* without its ``.fif`` / ``.fif.gz`` extension."""
    fname_out = Path(fname_out)
    name = fname_out.name
    for ext in (".fif.gz", ".fif"):
        if name.endswith(ext):
            return fname_out.with_name(name[: -len(ext)])
    return fname_out


def get_derivative_paths(fname_out):
    """Return the paths of all derivatives :class:`EEGPreprocessor` writes.

    File names follow BIDS: the entities in the file name of *fname_out*
    (``sub-``, ``ses-``, ``task-``, ``run-``, ...) are kept, its suffix
    (e.g. ``_eeg``) and any ``desc-`` entity are dropped, and each output
    gets a ``desc-`` label or its own suffix. The extension of *fname_out*
    (``.fif`` or ``.fif.gz``, default ``.fif``) is used for the FIF files.
    For ``<dir>/sub-01_task-walk_eeg.fif.gz``:

    - ``"clean"``: ``sub-01_task-walk_desc-clean_eeg.fif.gz``
    - ``"pre_ica"``: ``sub-01_task-walk_desc-preica_eeg.fif.gz``
    - ``"ica"``: ``sub-01_task-walk_ica.fif.gz``
    - ``"iclabels"``: ``sub-01_task-walk_iclabels.json``
    - ``"bad_channels"``: ``sub-01_task-walk_badchannels.json``
    - ``"dipoles"``: ``sub-01_task-walk_dipoles/`` (one file per IC)
    - ``"trans"``: ``sub-01_task-walk_trans.fif``
    - ``"report"``: ``sub-01_task-walk_report.html``

    All paths are in the folder of *fname_out*. The untouched input is not
    a derivative; it goes to the raw part of the BIDS dataset (see
    ``save_bids_raw`` in :class:`EEGPreprocessor`).

    Parameters
    ----------
    fname_out : str | Path
        The *fname_out* passed to :meth:`EEGPreprocessor.run_raw`.

    Returns
    -------
    paths : dict of str to Path
        The paths, keyed as listed above.
    """
    fname_out = Path(fname_out)
    name, ext = fname_out.name, ".fif"
    for _ext in (".fif.gz", ".fif"):
        if name.endswith(_ext):
            name, ext = name[: -len(_ext)], _ext
            break
    parts = name.split("_")
    if len(parts) > 1 and "-" not in parts[-1]:
        parts = parts[:-1]  # drop the suffix, e.g. "eeg"
    parts = [p for p in parts if not p.startswith("desc-")]
    if not parts:
        raise ValueError(f"Cannot build derivative names from {fname_out.name!r}.")
    base = "_".join(parts)
    folder = fname_out.parent
    return {
        "clean": folder / f"{base}_desc-clean_eeg{ext}",
        "pre_ica": folder / f"{base}_desc-preica_eeg{ext}",
        "ica": folder / f"{base}_ica{ext}",
        "iclabels": folder / f"{base}_iclabels.json",
        "bad_channels": folder / f"{base}_badchannels.json",
        "dipoles": folder / f"{base}_dipoles",
        "trans": folder / f"{base}_trans.fif",
        "report": folder / f"{base}_report.html",
    }


def _bids_raw_target(fname_out, bids_root=None):
    """Return ``(bids_root, entities)`` for writing the raw BIDS copy of a run.

    The BIDS entities (subject, session, task, run, acquisition) are parsed
    from the file name of *fname_out*, e.g.
    ``sub-P003_ses-S001_task-Walk_run-001_eeg.fif.gz``.

    If *bids_root* is ``None`` it is inferred as the folder that contains
    the ``derivatives`` directory *fname_out* lives in, e.g.
    ``<root>/derivatives/sub-P003/ses-S001/eeg/...`` gives ``<root>``.

    Raises
    ------
    ValueError
        If the root cannot be inferred or the file name lacks a subject or
        task entity.
    """
    from mne_bids import get_entities_from_fname

    fname_out = Path(fname_out)
    if bids_root is None:
        parts = fname_out.resolve().parts
        if "derivatives" not in parts:
            raise ValueError(
                f"Cannot infer the BIDS root: {fname_out} is not inside a "
                "'derivatives' folder. Pass bids_root=... to EEGPreprocessor."
            )
        bids_root = Path(*parts[: parts.index("derivatives")])
    ents = get_entities_from_fname(_output_stem(fname_out).name, on_error="ignore")
    entities = {
        key: ents.get(key)
        for key in ("subject", "session", "task", "run", "acquisition")
    }
    missing = [key for key in ("subject", "task") if not entities[key]]
    if missing:
        raise ValueError(
            f"Cannot parse BIDS {' and '.join(missing)} from file name "
            f"'{fname_out.name}' (expected e.g. 'sub-01_task-walk_eeg.fif.gz')."
        )
    return Path(bids_root), entities


def get_bad_chs(
    raw,
    pyprep_kwargs=None,
    notch_lines="europe",
    notch_width=1.0,
    line_noise_crit=None,
    deviation_threshold=3.5,
    ransac=None,
):
    """Detect bad EEG channels using PyPREP, FASTER, flatline, and line noise.

    Applies optional notch filtering and an average reference, then runs
    PyPREP's NoisyChannels (nan/flat, deviation, HF-noise, correlation,
    and optionally RANSAC), FASTER's channel-level correlation and variance on 1 s
    fixed-length epochs, and per-channel line-noise z-score detection.
    Returns a dictionary containing the union of all identified bad channels
    and per-method breakdowns.

    Parameters
    ----------
    raw : mne.io.Raw
        The MNE Raw object containing the EEG data. Channels listed in
        ``raw.info['bads']`` are treated as manual bads and will be included in
        the returned manual bad list and excluded from some computations.
    pyprep_kwargs : dict | None
        Keyword arguments passed to ``pyprep.find_noisy_channels.NoisyChannels``.
        If None, default settings are used. The optional key ``bad_by_manual``
        (list of channel names) can be provided to include manual bad channels.
    notch_lines : float | array-like | ``"europe"`` | ``"usa"`` | None
        The line frequencies for notch filtering. Strings ``"europe"`` and
        ``"usa"`` expand to 50/100/150 Hz and 60/120/180 Hz respectively.
        Defaults to ``"europe"``. Pass ``None`` to skip notch filtering.
    notch_width : float
        Width of the notch filter in Hz.
    line_noise_crit : float | None
        Z-score threshold for the per-channel line noise criterion.
        A channel is flagged as bad when its line-noise-to-broadband ratio
        exceeds this many standard deviations above the mean across channels.
        ``None`` (default) disables this criterion - recommended when ZapLine
        has already run, as residual line noise is negligible and the criterion
        may produce false rejections.
    deviation_threshold : float
        Z-score threshold for PyPREP's amplitude-deviation criterion
        (``find_bad_by_deviation``).  Channels whose robust z-score of
        channel-level RMS exceeds this value are flagged as bad.  Default
        ``3.5`` (tighter than PyPREP's built-in default of ``5.0``) to improve
        sensitivity on MoBI data where motion raises overall variance.
    ransac : bool | None
        Whether to run PyPREP's RANSAC bad-channel detection
        (``find_bad_by_ransac``), which uses spherical spline interpolation to
        predict each channel from a random subset of neighbours and flags
        channels that cannot be reconstructed.  Requires channel positions
        (i.e. a montage must be set on *raw*).

        - ``None`` (default): run RANSAC automatically when a montage is
          present; skip silently otherwise.
        - ``True``: always run; raises if no montage / positions available.
        - ``False``: never run.

    Returns
    -------
    bad_ch_dict : dict
        Dictionary with the following keys:

        - ``"all_bads"``: union of PyPREP, FASTER, line-noise, and manual bads.
        - ``"pyprep"``: dict from PyPREP's ``get_bads(as_dict=True)``.
        - ``"faster"``: dict from FASTER with a ``"bad_all"`` union key.
        - ``"bad_by_line_noise"``: channels flagged by per-channel line noise.
        - ``"bad_by_manual"``: manual bad channels.
    """
    raw = raw.copy()

    if isinstance(notch_lines, str):
        if notch_lines == "europe":
            notch_lines = np.arange(50, 151, 50)
        elif notch_lines == "usa":
            notch_lines = np.arange(60, 181, 60)
        else:
            raise ValueError(
                f"Unknown notch_lines preset: {notch_lines!r}. "
                "Use 'europe', 'usa', an array-like, or None."
            )

    # Consensus preprocessing steps before finding bads
    raw = raw.set_eeg_reference("average", projection=False)
    if notch_lines is not None:
        notch_lines_arr = np.asarray(notch_lines)
        nyquist = raw.info["sfreq"] / 2
        notch_lines_arr = notch_lines_arr[
            : np.searchsorted(notch_lines_arr, nyquist, side="right")
        ]
        raw.notch_filter(freqs=notch_lines_arr, notch_widths=notch_width)

    # === Per-channel line noise detection (before PyPREP / FASTER) ===
    bad_by_line_noise: list[str] = []
    if line_noise_crit is not None and notch_lines is not None:
        bad_by_line_noise = detect_bad_by_line_noise(
            raw,
            noise_freqs=np.asarray(notch_lines),
            z_thresh=float(line_noise_crit),
        )

    # === PyPREP ====
    # Merge user kwargs into the defaults (without mutating the caller's dict)
    pyprep_kwargs = {"reject_by_annotation": "omit", **(pyprep_kwargs or {})}

    bad_by_manual = list(pyprep_kwargs.get("bad_by_manual") or [])
    bad_by_manual = sorted(set(bad_by_manual) | set(raw.info["bads"]))
    pyprep_kwargs["bad_by_manual"] = bad_by_manual

    noisy_channels = NoisyChannels(raw, **pyprep_kwargs)
    noisy_channels.find_bad_by_nan_flat()
    noisy_channels.find_bad_by_deviation(deviation_threshold=deviation_threshold)
    noisy_channels.find_bad_by_hfnoise()
    noisy_channels.find_bad_by_correlation()

    _run_ransac = ransac if ransac is not None else (raw.get_montage() is not None)
    if _run_ransac:
        try:
            noisy_channels.find_bad_by_ransac()
        except Exception as _exc:
            LOGGER.warning(f"RANSAC bad-channel detection failed: {_exc}")

    bads_dict_pyprep = noisy_channels.get_bads(as_dict=True)

    # === FASTER ====
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True)
    picks = mne.pick_types(epochs.info, eeg=True, exclude=bad_by_manual)
    epochs.pick(picks)
    bads_dict_faster = mne_faster.find_bad_channels(
        epochs,
        return_by_metric=True,
        use_metrics=["correlation", "variance"],
    )
    bads_dict_faster["bad_all"] = list(
        {v for val in bads_dict_faster.values() if len(val) > 0 for v in val}
    )

    bad_chs = set()
    for bads_dict in [bads_dict_pyprep, bads_dict_faster]:
        for bad_chs_list in bads_dict.values():
            bad_chs.update(bad_chs_list)
    bad_chs.update(bad_by_manual)
    bad_chs.update(bad_by_line_noise)

    bad_ch_dict = {
        "all_bads": sorted(bad_chs),
        "pyprep": bads_dict_pyprep,
        "faster": bads_dict_faster,
        "bad_by_line_noise": bad_by_line_noise,
        "bad_by_manual": bad_by_manual,
    }
    return bad_ch_dict


# %% Classes


def _try_report(make_report_func, **kwargs):
    """Build the report; turn a failure into a warning and return ``None``."""
    try:
        return make_report_func(**kwargs)
    except Exception as exc:
        warnings.warn(
            f"Report generation failed ({type(exc).__name__}: {exc}); the "
            "other outputs are unaffected.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _expand_line_noise_freq(line_noise_freq, sfreq):
    """Return harmonics of *line_noise_freq* up to (but not exceeding) Nyquist.

    Parameters
    ----------
    line_noise_freq : float | ``"europe"`` | ``"usa"``
        Fundamental line-noise frequency in Hz, or a regional shortcut.
        ``"europe"`` → 50.0 Hz, ``"usa"`` → 60.0 Hz.
    sfreq : float
        Sampling frequency of the recording in Hz.

    Returns
    -------
    harmonics : numpy.ndarray
        1-D array of harmonic frequencies ``[f, 2f, 3f, ...]`` with all
        values strictly below Nyquist (``sfreq / 2``).
    """
    if isinstance(line_noise_freq, str):
        if line_noise_freq == "europe":
            base = 50.0
        elif line_noise_freq == "usa":
            base = 60.0
        else:
            raise ValueError(
                f"Unknown line_noise_freq preset: {line_noise_freq!r}. "
                "Use 'europe', 'usa', or a float (e.g. 50.0)."
            )
    else:
        base = float(line_noise_freq)
    nyquist = sfreq / 2
    # Leave a 3 Hz margin below Nyquist: mne_denoise's segment_data builds a
    # ±3 Hz bandpass around each harmonic, so a harmonic at exactly Nyquist
    # would produce a filter edge above fs/2 and crash scipy.signal.butter.
    n_harmonics = int((nyquist - 3.0) / base)
    return base * np.arange(1, n_harmonics + 1)


class EEGPreprocessor:
    """Preprocess EEG.

    preprocessing pipeline: ZapLine → bad channels → filter → ASR →
    ICA → dipole fitting → average re-reference → interpolate → save.

    Three recordings come out of the pipeline:

    - ``raw_source``: the input exactly as passed to :meth:`run_raw`
      (nothing done). In BIDS this is considered 'raw' format; it is
      written to the raw part of the BIDS dataset (``save_bids_raw``), not
      to the derivatives.
    - ``raw_pre_ica``: every step except ICA (renaming, pre-hook, line-noise
      removal, bad-channel detection, break annotation, bandpass, average
      reference projection, ASR). This is the data ICA is fitted on and
      applied to. Bad channels are marked but not yet interpolated, and the
      average reference is still an unapplied projection, so that
      :meth:`reapply_ica` can redo the final stage.
    - ``raw_clean``: everything done (ICA applied, average reference applied,
      bad channels interpolated).

    Parameters are listed in pipeline order.

    Parameters
    ----------
    loader : XDFLoader | None
        Configured loader used by :meth:`run` to read raw files.
        Not required when calling :meth:`run_raw` directly.
    channel_types : dict | None
        Channel name → MNE type mapping applied right after loading (only
        used by :meth:`run`).
    rename_channels : dict | str | None
        Channel renaming applied at the very start of :meth:`run_raw`.

        - ``dict``: explicit ``{old_name: new_name}`` mapping.
        - ``str``: strip this prefix from every channel name that starts
          with it (e.g. ``"BrainVision RDA_"``).
        - ``None`` (default): no renaming.
    pre_hook : callable | None
        Arbitrary transformation applied to the raw object **after** channel
        renaming and **before** any signal processing.

        The callable receives the ``mne.io.Raw`` object as its only argument
        and must return one of:

        - the modified ``raw`` object, or
        - a ``(raw, description)`` tuple, where *description* is a short
          string (≤ 120 characters) describing what the hook did.  The
          description is appended to the provenance metadata stored on the
          raw object and saved with the pipeline outputs.

        Use ``pre_hook`` for one-off operations that do not belong in the
        general pipeline but must happen before filtering, such as cropping
        the recording, injecting custom annotations, correcting a known
        hardware artefact, or converting units.  The hook runs before ZapLine,
        bad-channel detection, and all subsequent steps, so any changes it
        makes are seen by the entire pipeline.

        Example::

            def my_hook(raw):
                raw.crop(tmin=5.0)          # drop the first 5 s
                return raw, "cropped first 5 s"

            preprocessor = EEGPreprocessor(loader, pre_hook=my_hook)
    line_noise_freq : float | ``"europe"`` | ``"usa"``
        Fundamental line-noise frequency in Hz.  Harmonics are computed
        automatically up to (but not exceeding) the Nyquist frequency of the
        recording.  Accepted values:

        - ``float``: explicit fundamental (e.g. ``50.0`` or ``60.0``).
        - ``"europe"``: shortcut for 50 Hz (default).
        - ``"usa"``: shortcut for 60 Hz.

        The fundamental is passed to the line-noise removal step
        (when *line_noise_method* is not ``None``), which removes its harmonics
        itself; the full harmonic array is used by :func:`get_bad_chs` for
        notch-filtered bad-channel detection.
    line_noise_method : str | None
        DSS-based spectral cleaning algorithm applied before bandpass
        filtering.  ``None`` skips line-noise removal.  Default is
        ``"adaptive"`` (matching BeMoBIL).  One of:

        ``"adaptive"``
            ZapLine-plus (mne-denoise) with adaptive frequency detection.
        ``"zapline"``
            Standard ZapLine (mne-denoise), fixed-frequency.
        ``"dss_line"``
            Single-pass DSS (meegkit).
        ``"dss_line_iter"``
            Iterative DSS (meegkit).
    line_noise_kwargs : dict | None
        Extra keyword arguments forwarded to the backend of
        *line_noise_method* via :func:`remove_line`'s *method_kwargs*
        (e.g. ``{"threshold": 4.0}`` for ZapLine or ``{"nremove": 2}`` for
        ``"dss_line"``).  ``None`` uses defaults.
    get_bad_chs_kwargs : dict | None
        Extra keyword arguments forwarded to :func:`get_bad_chs`.  Supported
        keys (all optional):

        - ``"pyprep_kwargs"`` (*dict*): passed to PyPREP's
          :class:`~pyprep.NoisyChannels`; ``random_state`` is always
          overwritten with *rng_seed*.
        - ``"notch_width"`` (*float*, default ``1.0``): width of the notch
          filter used during bad-channel detection.
        - ``"line_noise_crit"`` (*float | None*, default ``None``): z-score
          threshold for the per-channel line-noise criterion; ``None``
          (default) disables this check - recommended when ZapLine has run.
        - ``"deviation_threshold"`` (*float*, default ``3.5``): z-score
          threshold for PyPREP's amplitude-deviation criterion.  Tighter
          than PyPREP's built-in default of ``5.0`` to improve sensitivity
          on MoBI data; raise to reduce false positives.
        - ``"ransac"`` (*bool | None*, default ``None``): run PyPREP RANSAC
          when ``None`` (auto) or ``True``; auto-detects from montage
          presence.  Set ``False`` to disable explicitly.
    annotate_breaks : bool
        If ``True``, run :func:`mne.preprocessing.annotate_break` to mark
        inter-block breaks (and other gaps between events) as ``BAD_break``
        annotations, which are then excluded (via ``reject_by_annotation``)
        from bad-channel detection, ICA fitting, and other downstream steps.
        Break detection can be overzealous on some recordings (e.g. sparse
        or irregular event structure), flagging most of the recording as
        "bad" even though the data itself is fine. Default ``False`` (skip
        this step entirely); set ``True`` to enable it, tuning behaviour via
        *annotate_break_kwargs* if needed.
    annotate_break_kwargs : dict | None
        Forwarded to :func:`mne.preprocessing.annotate_break`. Ignored when
        ``annotate_breaks=False``.
    filter_bands : tuple of float
        ``(l_freq, h_freq)`` for the main bandpass filter applied before
        ASR and ICA.
    asr : bool | dict
        Controls Artifact Subspace Reconstruction (ASR).

        - ``False`` (default): skip ASR; ``raw_pre_ica`` is the bandpassed,
          average-reference-projected data.
        - ``True``: run ASR with the default parameters of
          :func:`~bemobil_mne.preproc.utils.compute_asr`.
        - ``dict``: run ASR and pass the dict as keyword arguments to
          :func:`~bemobil_mne.preproc.utils.compute_asr` (e.g.
          ``{"cutoff": 10, "estimator": "lwf"}``).

        Either way, ``raw_pre_ica`` is the data ICA is applied to, and it is
        saved as ``*_desc-preica_eeg.fif`` so that ICA can be re-applied with
        a different set of excluded components via :meth:`reapply_ica`.
    filter_bands_ica : tuple of float
        ``(l_freq, h_freq)`` for the ICA-specific bandpass filter.
    downsample_ica : float | None
        Target sampling rate for ICA fitting.  ``None`` skips downsampling.
    ica_method : str
        ICA algorithm.  ``"jamica"`` (default) uses jamica, a JAX-accelerated
        reimplementation of AMICA, via :func:`jamica.fit_ica`, which returns
        a standard MNE ICA object directly; falls back to picard if the
        package is not installed.  Any other string is forwarded as the
        ``method`` argument to :class:`mne.preprocessing.ICA` (e.g.
        ``"picard"``, ``"fastica"``).  Ignored when ``fit_ica=False``.
    jamica_kwargs : dict | None
        Extra keyword arguments forwarded to :func:`jamica.fit_ica` when
        ``ica_method="jamica"``.  Useful for controlling convergence, e.g.
        ``{"max_iter": 2000}``.  ``None`` uses jamica's defaults.  Ignored
        when a non-jamica method is used or ``fit_ica=False``.
    fit_ica : bool
        If ``False``, skip ICA entirely (``raw_clean`` equals ``raw_pre_ica``).
    thresh : float
        ICLabel decision threshold.  Set to ``-1`` (default, matching
        BeMoBIL) to use **popularity-vote** mode: each IC is assigned to
        whichever class has the highest predicted probability; it is excluded
        if that class is not in *include_labels* (or is in *exclude_labels*).
        Any value in ``[0, 1]`` switches to **probability-threshold** mode:
        an IC is excluded only when its artifact-class probability meets or
        exceeds this value.
    exclude_labels : list of str | None
        ICLabel categories to exclude. Mutually exclusive with
        *include_labels*.
    include_labels : set of str | None
        ICLabel categories to keep; all others are excluded. ``None``
        (default) means all classes except ``"eye blink"`` (matching
        BeMoBIL's ``iclabel_classes = [1 2 4 5 6 7]``), unless
        *exclude_labels* is given. Mutually exclusive with
        *exclude_labels*.
    fit_dipoles : bool
        If ``True``, fit a dipole to each ICA component topography using the
        fsaverage BEM.  Requires a montage with digitisation.
    trans : mne.transforms.Transform | ``"fit"`` | ``"fsaverage"`` | None
        Head→MRI transform for dipole fitting. ``None`` and ``"fsaverage"``
        use the MNE built-in template; ``"fit"`` runs automatic coregistration.
        Ignored when ``fit_dipoles=False``.
    rv_thresh : float | None
        Residual-variance threshold for dipole fitting.  Components whose
        best-fitting dipole has RV >= *rv_thresh* are set to ``None`` in the
        ``dipoles`` and ``residuals`` output lists.  ``None`` keeps all
        dipoles.  Typical value: ``0.15`` (15 %).
    remove_outside_head : bool
        If ``True``, components whose dipole falls outside the head model
        (position norm > 0.13 m) are set to ``None`` in the output lists.
    rng_seed : int | None
        Random seed for ICA and PyPREP.
    event_id : dict | None
        Event map recorded in provenance metadata (no effect on processing).
    skip_if_exists : bool
        If ``True`` *and* ``overwrite=False``, skip the entire computation
        when the primary output file ``*_desc-clean_eeg.fif`` already
        exists and return the previously saved results instead.
    make_report : bool
        If ``True`` (default) and *fname_out* is provided, generate an
        :class:`mne.Report` summarising the preprocessing outputs and save it
        alongside the other derivatives as ``*_report.html``.
    save_bids_raw : bool
        If ``True`` (default) and *fname_out* is provided, also write
        ``raw_source`` (the untouched input) to the raw part of the BIDS
        dataset with :func:`mne_bids.write_raw_bids`, e.g.
        ``<bids_root>/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-..._eeg.vhdr``
        plus sidecars. Subject, session, task, run and acquisition are taken
        from the file name of *fname_out*. This is a one-time conversion:
        if the BIDS raw already exists it is left alone (see
        ``overwrite_bids_raw``). A failure here is reported as a warning and
        does not stop the pipeline.
    bids_root : str | Path | None
        Root of the BIDS dataset for ``save_bids_raw``. ``None`` (default)
        infers it as the folder containing the ``derivatives`` directory
        that *fname_out* is in.
    overwrite_bids_raw : bool
        If ``True``, replace an existing BIDS raw for this run. Independent of
        the ``overwrite`` argument of :meth:`run` / :meth:`run_raw`, which
        only affects the derivatives. Default ``False``.
    verbose : bool | str | int
        MNE verbosity level during processing.
    """

    def __init__(
        self,
        loader,
        *,
        channel_types: dict | None = None,
        rename_channels=None,
        pre_hook: object = None,
        line_noise_freq: float | str = "europe",
        line_noise_method: str | None = "adaptive",
        line_noise_kwargs: dict | None = None,
        get_bad_chs_kwargs: dict | None = None,
        annotate_breaks: bool = False,
        annotate_break_kwargs: dict | None = None,
        filter_bands: tuple[float | None, float | None] = (0.1, 100.0),
        asr: bool | dict = False,
        filter_bands_ica: tuple[float | None, float | None] = (1.75, None),
        downsample_ica: float | None = 250.0,
        ica_method: str = "jamica",
        jamica_kwargs: dict | None = None,
        fit_ica: bool = True,
        reject_by_annotation: bool = True,
        thresh: float = -1,
        exclude_labels: list | None = None,
        include_labels: set | None = None,
        fit_dipoles: bool = False,
        trans: object = None,
        rv_thresh: float | None = None,
        remove_outside_head: bool = False,
        rng_seed: int | None = None,
        event_id: dict | None = None,
        skip_if_exists: bool = False,
        make_report: bool = True,
        save_bids_raw: bool = True,
        bids_root: str | Path | None = None,
        overwrite_bids_raw: bool = False,
        verbose: bool | str | int = True,
    ):
        if not fit_ica and fit_dipoles:
            raise ValueError(
                "Cannot fit dipoles without fitting ICA. "
                "Set fit_ica=True or fit_dipoles=False."
            )

        if exclude_labels is not None and include_labels is not None:
            raise ValueError(
                "Specify either exclude_labels or include_labels, not both."
            )
        if exclude_labels is None and include_labels is None:
            include_labels = _DEFAULT_INCLUDE_LABELS

        # Validate line_noise_freq early so errors surface at construction time
        if isinstance(line_noise_freq, str) and line_noise_freq not in (
            "europe",
            "usa",
        ):
            raise ValueError(
                f"Unknown line_noise_freq preset: {line_noise_freq!r}. "
                "Use 'europe', 'usa', or a float (e.g. 50.0)."
            )

        self.loader = loader
        self.channel_types = channel_types
        self.rename_channels = rename_channels
        self.pre_hook = pre_hook
        self.line_noise_freq = line_noise_freq
        self.line_noise_method = line_noise_method
        self.line_noise_kwargs = line_noise_kwargs or {}
        self.get_bad_chs_kwargs = get_bad_chs_kwargs or {}
        self.annotate_breaks = annotate_breaks
        self.annotate_break_kwargs = annotate_break_kwargs
        self.filter_bands = filter_bands
        self.asr = asr
        self.filter_bands_ica = filter_bands_ica
        self.downsample_ica = downsample_ica
        self.ica_method = ica_method
        self.jamica_kwargs = jamica_kwargs
        self.reject_by_annotation = reject_by_annotation
        self.fit_ica = fit_ica
        self.thresh = thresh
        self.exclude_labels = exclude_labels
        self.include_labels = include_labels
        self.fit_dipoles = fit_dipoles
        self.trans = trans
        self.rv_thresh = rv_thresh
        self.remove_outside_head = remove_outside_head
        self.rng_seed = rng_seed
        self.event_id = event_id
        self.skip_if_exists = skip_if_exists
        self.make_report = make_report
        self.save_bids_raw = save_bids_raw
        self.bids_root = bids_root
        self.overwrite_bids_raw = overwrite_bids_raw
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public entry points

    def run(
        self,
        fname_in: str | Path,
        fname_out: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> tuple:
        """Load *fname_in* and run the full preprocessing pipeline.

        Uses :attr:`loader` (an :class:`~bemobil_mne.io.XDFLoader`), whose
        :meth:`~bemobil_mne.io.XDFLoader.load` returns a
        :class:`~bemobil_mne.io.MultimodalRecording`.  Only the Tier-1
        ``raw`` object is preprocessed; ``tier2`` is forwarded to
        :meth:`run_raw` for the report's drop-out plots, and ``events`` is
        discarded (call :meth:`run_raw` directly if you need it).

        The montage set by the loader is kept. Only when the loaded data
        has no montage is the 10-05 template applied (``colin27_1005``,
        named ``standard_1005`` before MNE 1.13); channels not in the
        template get a warning.

        Parameters
        ----------
        fname_in : str | Path
            Path to the raw input file (XDF or any MNE-readable format).
        fname_out : str | Path | None
            Where to save the derivatives, as a BIDS-style path such as
            ``<root>/derivatives/bemobil-mne/sub-01/eeg/sub-01_task-walk_eeg.fif``.
            Its entities name all outputs (see :func:`get_derivative_paths`),
            e.g. ``sub-01_task-walk_desc-clean_eeg.fif``. Pass ``None`` to
            skip saving.
        overwrite : bool
            Overwrite existing derivatives. Does not affect the BIDS raw
            (see ``overwrite_bids_raw``).

        Returns
        -------
        Same as :meth:`run_raw`.
        """
        rec = self.loader.load(fname_in)
        raw = rec.raw
        if self.channel_types:
            raw.set_channel_types(self.channel_types)
        if raw.get_montage() is None:
            # Fall back to standard 10-05 positions only when the loader did
            # not set a montage (e.g. XDFLoader(montage=...)).
            raw.set_montage(_template_montage_1005(), on_missing="warn")
        return self.run_raw(
            raw, fname_out=fname_out, overwrite=overwrite, tier2=rec.tier2
        )

    def run_raw(
        self,
        raw: mne.io.BaseRaw,
        fname_out: str | Path | None = None,
        *,
        overwrite: bool = False,
        tier2: dict | None = None,
    ) -> tuple:
        """Run the preprocessing pipeline on an already-loaded *raw* object.

        Channel types and montage must already be set by the caller.
        *raw* itself is not modified; the pipeline works on a copy.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Recording to preprocess.
        fname_out : str | Path | None
            Where to save the derivatives, as a BIDS-style path such as
            ``<root>/derivatives/bemobil-mne/sub-01/eeg/sub-01_task-walk_eeg.fif``.
            Its entities name all outputs (see :func:`get_derivative_paths`),
            e.g. ``sub-01_task-walk_desc-clean_eeg.fif``. Pass ``None`` to
            skip saving.
        overwrite : bool
            Overwrite existing derivatives. Does not affect the BIDS raw
            (see ``overwrite_bids_raw``).
        tier2 : dict | None
            Tier-2 streams from :class:`~bemobil_mne.io.MultimodalRecording`
            (e.g. ``rec.tier2``), kept at native rate and not merged into
            *raw*.  When provided and ``make_report=True``, each stream is
            plotted in full (decimated envelope, with drop-outs shaded) in
            its own report section.  ``None`` (default) skips this section.

        Returns
        -------
        raw_clean : mne.io.Raw
            Fully processed recording: ICA applied (and ASR, if enabled),
            average reference applied, bad channels interpolated.
        report : mne.Report | None
            Quality report (populated when ``make_report=True``, else ``None``).
        metadata : dict
            All other pipeline outputs, keyed by name:

            - ``raw_source`` (:class:`mne.io.Raw`): the input exactly as
              passed to :meth:`run_raw` (the same object), nothing done.
              Written to the BIDS raw folder when ``save_bids_raw=True``;
              when loaded from cache, read back from there (``None`` if
              not found).
            - ``raw_pre_ica`` (:class:`mne.io.Raw`): every step except ICA
              (line-noise removal, bandpass, ASR if enabled). The data ICA
              is fitted on. Bad channels are marked but not interpolated and
              the average reference is still an unapplied projection. Saved
              as ``*_desc-preica_eeg.fif``; input to :meth:`reapply_ica`.
            - ``ica`` (:class:`mne.preprocessing.ICA`): the fitted ICA, with
              ``ica.exclude`` set from the ICLabel classification. An
              unfitted stub when ``fit_ica=False``.
            - ``ic_labels`` (``dict``): ICLabel output per component
              (``"labels"`` and ``"y_pred_proba"``). Empty when
              ``fit_ica=False``.
            - ``dipoles`` (``list`` of :class:`mne.Dipole` | ``None``): one
              equivalent current dipole per IC, ``None`` where removed by
              ``rv_thresh`` or ``remove_outside_head``. Empty unless
              ``fit_dipoles=True``.
            - ``residuals`` (``list`` of :class:`mne.Evoked` | ``None``):
              residual field of each dipole fit, aligned with ``dipoles``.
            - ``trans`` (:class:`mne.transforms.Transform` | ``None``): the
              head-to-MRI transform used for dipole fitting. ``None`` unless
              ``fit_dipoles=True``.
            - ``bad_ch_dict`` (``dict``): detected bad channels.
              ``"all_bads"`` is the combined list; ``"pyprep"``,
              ``"faster"``, ``"bad_by_line_noise"`` and ``"bad_by_manual"``
              hold the per-method results.
        """
        old_verbose = mne.set_log_level(verbose=self.verbose, return_old_level=True)
        try:
            return self._run_raw(
                raw, fname_out=fname_out, overwrite=overwrite, tier2=tier2
            )
        finally:
            mne.set_log_level(verbose=old_verbose)

    def _run_raw(self, raw, fname_out, *, overwrite, tier2):
        """Run the pipeline body; see :meth:`run_raw`."""
        timer = StepTimer()

        # --- Skip-if-exists caching ---
        if (
            self.skip_if_exists
            and not overwrite
            and fname_out is not None
            and get_derivative_paths(fname_out)["clean"].exists()
        ):
            import logging as _log

            _log.getLogger(__name__).info(
                f"skip_if_exists=True: loading cached outputs from {fname_out}"
            )
            return self._load_cached_outputs(fname_out)

        # --- Source (nothing done) ---
        # The caller's object is never modified: keep it as raw_source and
        # run every step on a preloaded copy.
        raw_source = raw
        raw = raw.copy().load_data()

        # --- Provenance ---
        filenames = [str(p) for p in getattr(raw, "filenames", []) if p]
        src = filenames[0] if len(filenames) == 1 else (filenames or None)
        set_descriptor(raw, init_descriptor(src, pipeline="EEGPreprocessor.run_raw"))

        # --- Channel renaming ---
        if self.rename_channels is not None:
            if isinstance(self.rename_channels, dict):
                raw.rename_channels(self.rename_channels)
                append_desc(raw, name="rename_channels", mapping=self.rename_channels)
            elif isinstance(self.rename_channels, str):
                prefix = self.rename_channels
                mapping = {
                    ch: ch[len(prefix) :]
                    for ch in raw.ch_names
                    if ch.startswith(prefix)
                }
                if mapping:
                    raw.rename_channels(mapping)
                append_desc(raw, name="rename_channels", strip_prefix=prefix)
            else:
                raise TypeError(
                    "rename_channels must be a dict or str, got"
                    f" {type(self.rename_channels)!r}"
                )

        # --- pre_hook ---
        pre_hook_description = None
        pre_hook_source = None
        if self.pre_hook is not None:
            if not callable(self.pre_hook):
                raise TypeError("`pre_hook` must be callable.")
            try:
                import inspect as _inspect

                pre_hook_source = _inspect.getsource(self.pre_hook)
            except (OSError, TypeError):
                pass
            result = self.pre_hook(raw)
            if isinstance(result, tuple):
                raw, pre_hook_description = result
            else:
                raw = result
            if pre_hook_description:
                append_desc(raw, name="pre_hook", description=pre_hook_description)

        # --- Line-noise removal ---
        if self.line_noise_method is not None:
            t0 = time.perf_counter()
            raw = remove_line(
                raw,
                noise_freqs=self.line_noise_freq,
                method=self.line_noise_method,
                method_kwargs=self.line_noise_kwargs,
            )
            append_desc(
                raw,
                name="remove_line",
                method=self.line_noise_method,
                noise_freqs=self.line_noise_freq,
                method_kwargs=self.line_noise_kwargs,
            )
            timer.log_step("remove_line", time.perf_counter() - t0)

        # --- Bad channel detection ---
        t0 = time.perf_counter()
        _bad_ch_kw = dict(self.get_bad_chs_kwargs)
        _pyprep_kw = _bad_ch_kw.pop("pyprep_kwargs", {})
        _pyprep_kw["random_state"] = self.rng_seed
        _notch_lines = _expand_line_noise_freq(self.line_noise_freq, raw.info["sfreq"])
        bad_ch_dict = get_bad_chs(
            raw,
            pyprep_kwargs=_pyprep_kw,
            notch_lines=_notch_lines,
            **_bad_ch_kw,
        )
        eeg_chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
        if not set(eeg_chs) - set(bad_ch_dict["all_bads"]):
            raise RuntimeError(
                f"All {len(eeg_chs)} EEG channels were flagged as bad "
                f"(see bad_ch_dict: pyprep={bad_ch_dict['pyprep'].get('bad_all')}, "
                f"faster={bad_ch_dict['faster'].get('bad_all')}). Check the "
                "recording, or relax the criteria via get_bad_chs_kwargs "
                "(e.g. deviation_threshold, ransac=False)."
            )
        raw.info["bads"] = bad_ch_dict["all_bads"]
        append_desc(
            raw,
            name="bad_channel_detection",
            all_bads=bad_ch_dict["all_bads"],
        )
        timer.log_step("bad_channel_detection", time.perf_counter() - t0)

        # --- Annotate breaks (optional; off by default) ---
        t0 = time.perf_counter()
        if self.annotate_breaks:
            annots_break, final_break_kwargs = _annotate_break_iter(
                raw, self.annotate_break_kwargs
            )
            raw.set_annotations(raw.annotations + annots_break)
            append_desc(raw, name="annotate_breaks", **final_break_kwargs)
        timer.log_step("annotate_breaks", time.perf_counter() - t0)

        # --- Bandpass + avg-ref projection ---
        t0 = time.perf_counter()
        raw.filter(l_freq=self.filter_bands[0], h_freq=None)
        append_desc(
            raw,
            name="highpass",
            **sig_params(mne.io.Raw.filter, l_freq=self.filter_bands[0], h_freq=None),
        )
        _h_freq = self.filter_bands[1]
        if _h_freq is not None:
            _nyquist = raw.info["sfreq"] / 2
            if _h_freq >= _nyquist:
                _h_freq = _nyquist * 0.99
                warnings.warn(
                    "filter_bands h_freq clipped to "
                    f"{_h_freq:.2f} Hz (Nyquist = {_nyquist:.1f} Hz)",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raw.filter(l_freq=None, h_freq=_h_freq)
        append_desc(
            raw,
            name="lowpass",
            **sig_params(mne.io.Raw.filter, l_freq=None, h_freq=_h_freq),
        )
        raw.set_eeg_reference(ref_channels="average", projection=True)
        append_desc(raw, name="avg_ref_projection")
        timer.log_step("bandpass_avg_ref", time.perf_counter() - t0)

        # --- ASR ---
        t0 = time.perf_counter()
        if self.asr is False:
            raw_pre_ica = raw
        else:
            _asr_kwargs = self.asr if isinstance(self.asr, dict) else {}
            raw_pre_ica = compute_asr(raw, **_asr_kwargs)
            append_desc(raw_pre_ica, name="asr", **_asr_kwargs)
        timer.log_step("asr", time.perf_counter() - t0)

        # --- ICA ---
        dipoles, residuals = [], []
        trans_out = None
        # Snapshot annotations present at ICA time (after break annotation +
        # bad-channel detection, before ICA filtering/subsampling removes them)
        ica_annots = raw_pre_ica.annotations.copy()
        t0 = time.perf_counter()

        if self.fit_ica:
            ica, ic_labels = compute_ica(
                raw_pre_ica,
                filter_bands_ica=self.filter_bands_ica,
                downsample_ica=self.downsample_ica,
                thresh=self.thresh,
                rng_seed=self.rng_seed,
                exclude_labels=self.exclude_labels,
                include_labels=self.include_labels,
                ica_method=self.ica_method,
                jamica_kwargs=self.jamica_kwargs,
                reject_by_annotation=self.reject_by_annotation,
            )
            append_desc(
                raw_pre_ica,
                name="ica",
                method=self.ica_method,
                n_excluded=len(ica.exclude),
                excluded=ica.exclude,
                thresh=self.thresh,
            )
        else:
            warnings.warn(
                "fit_ica=False: raw_clean will NOT be ICA cleaned.",
                RuntimeWarning,
                stacklevel=2,
            )
            ica = mne.preprocessing.ICA(
                n_components=None,
                random_state=self.rng_seed,
                method="picard",
                fit_params={"ortho": False, "extended": True},
            )  # stub - never fitted
            ic_labels = {}

        timer.log_step("ica", time.perf_counter() - t0)

        # --- Dipole fitting ---
        if self.fit_dipoles:
            t0 = time.perf_counter()
            trans_out = _handle_trans(self.trans, raw_pre_ica.info)
            dipoles, residuals = fit_dipoles_on_ica(
                ica,
                raw_pre_ica.info,
                trans_out,
                rv_thresh=self.rv_thresh,
                remove_outside_head=self.remove_outside_head,
            )
            n_valid = sum(d is not None for d in dipoles)
            append_desc(
                raw_pre_ica,
                name="dipole_fitting",
                n_dipoles=len(dipoles),
                n_dipolar=n_valid,
                rv_thresh=self.rv_thresh,
                remove_outside_head=self.remove_outside_head,
            )
            timer.log_step("fit_dipoles", time.perf_counter() - t0)

        # --- Apply ICA, avg-ref, interpolate ---
        t0 = time.perf_counter()
        if self.fit_ica:
            raw_clean = ica.apply(raw_pre_ica.copy())
        else:
            raw_clean = raw_pre_ica.copy()
        # Bake in the average EEG reference here, at the very end of the
        # pipeline. Up to this point the reference has only ever existed as
        # an unapplied SSP projection (added with projection=True in the
        # bandpass stage and inherited via .copy()/ICA through
        # raw_pre_ica -> raw_clean). It is only actually applied to the data
        # now, once, via apply_proj(). raw_pre_ica keeps it unapplied so
        # reapply_ica() can repeat this stage. The `set_eeg_reference` call
        # is only a defensive fallback in case raw_clean somehow doesn't
        # already carry the projection.
        if not any(
            p["desc"] == "Average EEG reference" for p in raw_clean.info["projs"]
        ):
            raw_clean.set_eeg_reference(ref_channels="average", projection=True)
        raw_clean.apply_proj()
        append_desc(raw_clean, name="avg_ref")
        # Exclude EOG-typed channels from interpolation (they are not on the
        # scalp and spherical interpolation is not meaningful for them)
        eog_chs = [
            ch
            for ch, d in zip(raw_clean.ch_names, raw_clean.info["chs"])
            if d["kind"] == mne.io.constants.FIFF.FIFFV_EOG_CH
        ]
        bads_to_interp = [b for b in raw_clean.info["bads"] if b not in eog_chs]
        raw_clean.info["bads"] = bads_to_interp
        raw_clean.interpolate_bads(reset_bads=True, method="spline")
        append_desc(
            raw_clean,
            name="interpolate_bads",
            method="spline",
            interpolated=bads_to_interp,
            eog_excluded=eog_chs,
        )

        timer.log_step("rereference_interpolate", time.perf_counter() - t0)

        # --- Save (optional) ---
        if fname_out is not None:
            t0 = time.perf_counter()
            self._save_outputs(
                fname_out,
                raw_clean=raw_clean,
                raw_pre_ica=raw_pre_ica,
                ica=ica,
                ic_labels=ic_labels,
                dipoles=dipoles,
                residuals=residuals,
                trans=trans_out,
                bad_ch_dict=bad_ch_dict,
                overwrite=overwrite,
            )
            timer.log_step("save", time.perf_counter() - t0)

            # --- Untouched input -> raw BIDS folder (optional) ---
            if self.save_bids_raw:
                t0 = time.perf_counter()
                # Uses self.overwrite_bids_raw, NOT `overwrite`: rerunning
                # the derivatives must not rewrite the BIDS raw.
                self._save_bids_raw(raw_source, fname_out)
                timer.log_step("save_bids_raw", time.perf_counter() - t0)

        # --- Report (optional) ---
        # Built after saving so a failing report never costs the derivatives.
        report = None
        if self.make_report:
            from bemobil_mne.preproc.make_report import make_report as _make_report

            report = _try_report(
                _make_report,
                raw_pre_ica=raw_pre_ica,
                raw_clean=raw_clean,
                ica=ica,
                ic_labels=ic_labels,
                dipoles=dipoles,
                residuals=residuals,
                trans=trans_out,
                bad_ch_dict=bad_ch_dict,
                fname_out=fname_out,
                event_id=self.event_id,
                thresh=self.thresh,
                step_timings=timer.timings,
                tier2=tier2,
                pre_hook_description=pre_hook_description,
                pre_hook_source=pre_hook_source,
                ica_annots=ica_annots,
                raw_source=raw_source,
            )
            if report is not None and fname_out is not None:
                self._save_report(report, fname_out, overwrite=overwrite)

        import logging as _logging

        _logging.getLogger(__name__).info(timer.format_summary())

        metadata = {
            "raw_source": raw_source,
            "raw_pre_ica": raw_pre_ica,
            "ica": ica,
            "ic_labels": ic_labels,
            "dipoles": dipoles,
            "residuals": residuals,
            "trans": trans_out,
            "bad_ch_dict": bad_ch_dict,
        }
        return (raw_clean, report, metadata)

    # ------------------------------------------------------------------
    # Private helpers

    def _save_outputs(
        self,
        fname_out,
        *,
        raw_clean,
        raw_pre_ica,
        ica,
        ic_labels,
        dipoles,
        residuals,
        trans,
        bad_ch_dict,
        overwrite,
    ):
        """Save all pipeline derivatives to disk (see get_derivative_paths)."""
        paths = get_derivative_paths(fname_out)
        paths["clean"].parent.mkdir(parents=True, exist_ok=True)

        # raw_clean is the primary output and the skip_if_exists sentinel
        raw_clean.save(paths["clean"], overwrite=overwrite, verbose="error")

        # Pre-ICA data (ASR-cleaned if ASR ran, avg-ref still an unapplied
        # projection). Also the input to reapply_ica().
        raw_pre_ica.save(paths["pre_ica"], overwrite=overwrite, verbose="error")
        if self.fit_ica and ica.current_fit != "unfitted":
            ica.save(paths["ica"], overwrite=overwrite, verbose="error")
            with open(paths["iclabels"], "w") as f:
                json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)

        with open(paths["bad_channels"], "w") as f:
            json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)

        if dipoles:
            dipdir = paths["dipoles"]
            dipdir.mkdir(parents=True, exist_ok=True)
            for i, dip in enumerate(dipoles):
                if dip is None:
                    continue
                dip.save(dipdir / f"ic-{i:02}-dip.bdip", overwrite=overwrite)
            for i, residual in enumerate(residuals):
                if residual is None:
                    continue
                residual.save(
                    dipdir / f"ic-{i:02}-residual.fif.gz",
                    overwrite=overwrite,
                    verbose="error",
                )

        if dipoles and trans is not None:
            trans.save(paths["trans"], overwrite=overwrite, verbose="error")

    @staticmethod
    def _save_report(report, fname_out, *, overwrite):
        """Save *report* next to the derivatives of *fname_out*."""
        path = get_derivative_paths(fname_out)["report"]
        path.parent.mkdir(parents=True, exist_ok=True)
        report.save(str(path), overwrite=overwrite, open_browser=False)

    def _save_bids_raw(self, raw_source, fname_out):
        """Write the untouched input to the raw part of the BIDS dataset.

        Returns the written :class:`mne_bids.BIDSPath`, or ``None`` when the
        file already exists (and :attr:`overwrite_bids_raw` is ``False``) or
        writing failed.
        Failures are turned into warnings so the derivatives already saved
        are not lost in a batch run.
        """
        from mne_bids import BIDSPath

        from bemobil_mne.io.bids_export import export_to_bids

        try:
            bids_root, ents = _bids_raw_target(fname_out, self.bids_root)
        except ValueError as exc:
            warnings.warn(f"save_bids_raw skipped: {exc}", stacklevel=3)
            return None

        target = BIDSPath(root=bids_root, datatype="eeg", suffix="eeg", **ents)
        existing = sorted(target.directory.glob(target.basename + ".*"))
        if existing and not self.overwrite_bids_raw:
            LOGGER.info(
                f"save_bids_raw: {existing[0]} already exists, not overwriting "
                "(set overwrite_bids_raw=True to replace it)."
            )
            return None

        lnf = self.line_noise_freq
        line_freq = {"europe": 50.0, "usa": 60.0}[lnf] if isinstance(lnf, str) else lnf

        try:
            bids_path = export_to_bids(
                raw_source,
                bids_root=bids_root,
                line_freq=float(line_freq),
                overwrite=self.overwrite_bids_raw,
                verbose=False,
                **ents,
            )
        except Exception as exc:
            warnings.warn(
                f"save_bids_raw failed for {target.basename}: "
                f"{type(exc).__name__}: {exc}",
                stacklevel=3,
            )
            return None
        LOGGER.info(f"save_bids_raw: wrote {bids_path.fpath}")
        return bids_path

    def _read_bids_raw(self, fname_out):
        """Read the untouched input back from the raw part of the BIDS dataset.

        Returns ``None`` (and logs why) when the BIDS raw cannot be found.
        """
        from mne_bids import BIDSPath, read_raw_bids

        try:
            bids_root, ents = _bids_raw_target(fname_out, self.bids_root)
        except ValueError as exc:
            LOGGER.info(f"No BIDS raw for {Path(fname_out).name}: {exc}")
            return None
        target = BIDSPath(root=bids_root, datatype="eeg", suffix="eeg", **ents)
        for ext in (".vhdr", ".edf", ".bdf", ".set"):
            if target.copy().update(extension=ext).fpath.exists():
                return read_raw_bids(target.update(extension=ext), verbose=False)
        LOGGER.info(f"No BIDS raw found at {target.directory / target.basename}.*")
        return None

    def _load_cached_outputs(self, fname_out):
        """Load previously saved pipeline outputs and return the run_raw 3-tuple.

        Used by the ``skip_if_exists`` fast-path in :meth:`run_raw` and by
        :meth:`reapply_ica`. Missing auxiliary files yield ``None`` / empty
        collections. Recordings are not preloaded. ``raw_source`` is read
        from the raw part of the BIDS dataset.
        """
        paths = get_derivative_paths(fname_out)

        raw_clean = mne.io.read_raw(paths["clean"], verbose="error")
        raw_source = self._read_bids_raw(fname_out)

        _pre_ica_path = paths["pre_ica"]
        raw_pre_ica = (
            mne.io.read_raw(_pre_ica_path, verbose="error")
            if _pre_ica_path.exists()
            else None
        )

        _ica_path = paths["ica"]
        ica = (
            mne.preprocessing.read_ica(_ica_path, verbose="error")
            if _ica_path.exists()
            else mne.preprocessing.ICA(method="picard")
        )

        _labels_path = paths["iclabels"]
        ic_labels: dict = {}
        if _labels_path.exists():
            with open(_labels_path) as f:
                ic_labels = json.load(f)

        _bad_path = paths["bad_channels"]
        bad_ch_dict: dict = {}
        if _bad_path.exists():
            with open(_bad_path) as f:
                bad_ch_dict = json.load(f)

        # Dipoles / residuals are saved one file per IC, skipping ICs whose
        # fit was rejected (None). Rebuild lists indexed by IC number so that
        # dipoles[i] belongs to IC i, with None in the gaps.
        _dipdir = paths["dipoles"]
        dipoles: list = []
        residuals: list = []
        if _dipdir.exists():
            dip_files = {
                int(p.name.split("-")[1]): p for p in _dipdir.glob("ic-*-dip.bdip")
            }
            res_files = {
                int(p.name.split("-")[1]): p
                for p in _dipdir.glob("ic-*-residual.fif.gz")
            }
            n_ics = getattr(ica, "n_components_", None) or 0
            n = max([n_ics, *(i + 1 for i in dip_files), *(i + 1 for i in res_files)])
            dipoles = [
                mne.read_dipole(dip_files[i], verbose="error")
                if i in dip_files
                else None
                for i in range(n)
            ]
            residuals = [
                mne.read_evokeds(res_files[i], verbose="error")[0]
                if i in res_files
                else None
                for i in range(n)
            ]

        _trans_path = paths["trans"]
        trans = (
            mne.transforms.read_trans(_trans_path, verbose="error")
            if _trans_path.exists()
            else None
        )

        metadata = {
            "raw_source": raw_source,
            "raw_pre_ica": raw_pre_ica,
            "ica": ica,
            "ic_labels": ic_labels,
            "dipoles": dipoles,
            "residuals": residuals,
            "trans": trans,
            "bad_ch_dict": bad_ch_dict,
        }
        return (raw_clean, None, metadata)

    # ------------------------------------------------------------------
    # Post-hoc ICA editing

    def reapply_ica(
        self,
        fname_out: str | Path,
        *,
        exclude: list[int] | None = None,
        add: list[int] | tuple = (),
        remove: list[int] | tuple = (),
        save: bool = True,
        make_report: bool | None = None,
    ) -> tuple:
        """Re-apply a saved ICA with a manually edited set of excluded ICs.

        Loads the outputs of a previous :meth:`run_raw` call from disk,
        changes ``ica.exclude`` and repeats only the final pipeline stage
        (ICA application, average reference, bad-channel interpolation).
        Bad-channel detection, ASR, ICA fitting and dipole fitting are not
        re-run.

        Parameters
        ----------
        fname_out : str | Path
            The *fname_out* used for the original run.
        exclude : list of int | None
            Full list of ICs to exclude, replacing the automatic selection.
            If ``None``, start from the saved ``ica.exclude`` and apply
            *add* / *remove*.
        add : list of int
            ICs to exclude in addition to the saved ones. Ignored when
            *exclude* is given.
        remove : list of int
            ICs to keep even though they were excluded before. Ignored when
            *exclude* is given.
        save : bool
            If ``True`` (default), overwrite the clean recording
            (``*_desc-clean_eeg.fif``), the ICA file (``*_ica.fif``) and,
            if made, the report.
        make_report : bool | None
            Regenerate the HTML report. ``None`` (default) uses
            :attr:`make_report`.

        Returns
        -------
        Same 3-tuple as :meth:`run_raw`: ``(raw_clean, report, metadata)``.
        """
        paths = get_derivative_paths(fname_out)
        if not paths["pre_ica"].exists():
            raise FileNotFoundError(
                f"No pre-ICA recording ({paths['pre_ica'].name}) found in "
                f"{paths['pre_ica'].parent}. Re-run the pipeline with "
                "fit_ica=True to create it."
            )

        _, _, metadata = self._load_cached_outputs(fname_out)
        ica = metadata["ica"]
        if ica.current_fit == "unfitted":
            raise RuntimeError(f"No fitted ICA found for {fname_out}.")

        n_ics = ica.n_components_
        auto_exclude = sorted(int(i) for i in ica.exclude)
        if exclude is not None:
            new_exclude = set(exclude)
        else:
            new_exclude = (set(auto_exclude) | set(add)) - set(remove)
        bad_idx = [i for i in new_exclude if not 0 <= i < n_ics]
        if bad_idx:
            raise ValueError(f"IC indices out of range 0..{n_ics - 1}: {bad_idx}")
        ica.exclude = sorted(int(i) for i in new_exclude)

        raw_pre_ica = metadata["raw_pre_ica"].load_data()
        raw_clean = ica.apply(raw_pre_ica.copy())

        # Same final stage as run_raw: bake in the avg-ref projection,
        # then interpolate bad (non-EOG) channels.
        if not any(
            p["desc"] == "Average EEG reference" for p in raw_clean.info["projs"]
        ):
            raw_clean.set_eeg_reference(ref_channels="average", projection=True)
        raw_clean.apply_proj()
        append_desc(raw_clean, name="avg_ref")
        eog_chs = [
            ch
            for ch, d in zip(raw_clean.ch_names, raw_clean.info["chs"])
            if d["kind"] == mne.io.constants.FIFF.FIFFV_EOG_CH
        ]
        bads_to_interp = [b for b in raw_clean.info["bads"] if b not in eog_chs]
        raw_clean.info["bads"] = bads_to_interp
        raw_clean.interpolate_bads(reset_bads=True, method="spline")
        append_desc(
            raw_clean,
            name="interpolate_bads",
            method="spline",
            interpolated=bads_to_interp,
            eog_excluded=eog_chs,
        )
        append_desc(
            raw_clean,
            name="manual_ic_rejection",
            auto_excluded=auto_exclude,
            excluded=ica.exclude,
        )

        if save:
            raw_clean.save(paths["clean"], overwrite=True, verbose="error")
            ica.save(paths["ica"], overwrite=True, verbose="error")

        report = None
        if self.make_report if make_report is None else make_report:
            from bemobil_mne.preproc.make_report import make_report as _make_report

            raw_source = metadata["raw_source"]
            report = _try_report(
                _make_report,
                raw_pre_ica=raw_pre_ica,
                raw_clean=raw_clean,
                ica=ica,
                ic_labels=metadata["ic_labels"],
                dipoles=metadata["dipoles"],
                residuals=metadata["residuals"],
                trans=metadata["trans"],
                bad_ch_dict=metadata["bad_ch_dict"],
                fname_out=fname_out,
                event_id=self.event_id,
                thresh=self.thresh,
                ica_annots=raw_pre_ica.annotations,
                raw_source=(raw_source.load_data() if raw_source is not None else None),
            )
            if save and report is not None:
                self._save_report(report, fname_out, overwrite=True)

        return (raw_clean, report, metadata)
