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
    compute_zapline,
    detect_bad_by_line_noise,
    fit_dipoles_on_ica,
    get_raw_subset,
    init_descriptor,
    set_descriptor,
    sig_params,
)

# Keep NumpyEncoder importable from this module for backward compatibility
NumpyEncoder = _NumpyEncoder

LOGGER = logging.getLogger(__name__)

# %% Functions


def get_bad_chs(
    raw,
    pyprep_kwargs=None,
    notch_lines=np.arange(50, 151, 50),
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
        Pass ``None`` to skip notch filtering.
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
    raw = raw.set_eeg_reference("average", projection=True)
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
    default_pyprep_kwargs = {"reject_by_annotation": "omit"}
    if pyprep_kwargs is not None:
        default_pyprep_kwargs.update(pyprep_kwargs)
    else:
        pyprep_kwargs = default_pyprep_kwargs

    bad_by_manual = pyprep_kwargs.get("bad_by_manual", [])
    bad_by_manual = list(set(bad_by_manual + raw.info["bads"]))
    pyprep_kwargs.update({"bad_by_manual": bad_by_manual})

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
        set(v for val in bads_dict_faster.values() if len(val) > 0 for v in val)
    )

    bad_chs = set()
    for bads_dict in [bads_dict_pyprep, bads_dict_faster]:
        for bad_chs_list in bads_dict.values():
            bad_chs.update(bad_chs_list)
    bad_chs.update(bad_by_manual)
    bad_chs.update(bad_by_line_noise)

    bad_ch_dict = {
        "all_bads": list(bad_chs),
        "pyprep": bads_dict_pyprep,
        "faster": bads_dict_faster,
        "bad_by_line_noise": bad_by_line_noise,
        "bad_by_manual": bad_by_manual,
    }

    raw.del_proj()
    return bad_ch_dict


# %% Classes


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

        The resulting harmonic array is used both by the ZapLine spectral
        cleaning step (when *zapline_method* is not ``None``) and by
        :func:`get_bad_chs` for notch-filtered bad-channel detection.
    zapline_method : str | None
        DSS-based spectral cleaning algorithm applied before bandpass
        filtering.  ``None`` skips ZapLine entirely.  Default is
        ``"adaptive"`` (matching BeMoBIL).  One of:

        ``"adaptive"``
            ZapLine-plus (mne-denoise) with adaptive frequency detection.
        ``"zapline"``
            Standard ZapLine (mne-denoise), fixed-frequency.
        ``"dss_line"``
            Single-pass DSS (meegkit).
        ``"dss_line_iter"``
            Iterative DSS (meegkit).
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
        ``(l_freq, h_freq)`` for the main bandpass filter applied to
        ``raw_minimal``.
    subset_chs : list of str | None
        Channels for the ``raw_subset`` output (minimally processed, without
        average reference).  Defaults to some central and frontal channels when
        ``None`` but produces ``None`` if none are found in the data.
    asr : bool | dict
        Controls Artifact Subspace Reconstruction (ASR).

        - ``False`` (default): skip ASR; ``raw_asr`` is a copy of
          ``raw_minimal``.
        - ``True``: run ASR with the default parameters of
          :func:`~bemobil_mne.preproc.utils.compute_asr`.
        - ``dict``: run ASR and pass the dict as keyword arguments to
          :func:`~bemobil_mne.preproc.utils.compute_asr` (e.g.
          ``{"cutoff": 10, "estimator": "lwf"}``).
    filter_bands_ica : tuple of float
        ``(l_freq, h_freq)`` for the ICA-specific bandpass filter.
    downsample_ica : float | None
        Target sampling rate for ICA fitting.  ``None`` skips downsampling.
    ica_method : str
        ICA algorithm.  ``"amica"`` (default) uses AMICA via
        ``amica-python`` and converts to MNE ICA; falls back to picard if the
        package is not installed.  Any other string is forwarded as the
        ``method`` argument to :class:`mne.preprocessing.ICA` (e.g.
        ``"picard"``, ``"fastica"``).  Ignored when ``fit_ica=False``.
    amica_kwargs : dict | None
        Extra keyword arguments forwarded to :class:`amica.AMICA` when
        ``ica_method="amica"``.  Useful for controlling convergence, e.g.
        ``{"max_iter": 2000}``.  ``None`` uses AMICA defaults.  Ignored
        when a non-AMICA method is used or ``fit_ica=False``.
    fit_ica : bool
        If ``False``, skip ICA entirely (``raw_clean`` equals ``raw_asr``).
    thresh : float
        ICLabel decision threshold.  Set to ``-1`` (default, matching
        BeMoBIL) to use **popularity-vote** mode: each IC is assigned to
        whichever class has the highest predicted probability; it is excluded
        if that class is not in *include_labels* (or is in *exclude_labels*).
        Any value in ``[0, 1]`` switches to **probability-threshold** mode:
        an IC is excluded only when its artifact-class probability meets or
        exceeds this value.
    exclude_labels : list of str | None
        ICLabel categories to exclude.  Mutually exclusive with
        *include_labels*.
    include_labels : set of str | None
        ICLabel categories to keep; all others are excluded.  Defaults to
        all classes except ``"eye blink"`` (matching BeMoBIL's
        ``iclabel_classes = [1 2 4 5 6 7]``).  Mutually exclusive with
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
        when the primary output file ``{fname_out}_clean.fif.gz`` already
        exists and return the previously saved results instead.
    make_report : bool
        If ``True`` (default) and *fname_out* is provided, generate an
        :class:`mne.Report` summarising the preprocessing outputs and save it
        alongside the other derivatives as ``{fname_out}_report.html``.
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
        zapline_method: str | None = "adaptive",
        get_bad_chs_kwargs: dict | None = None,
        annotate_breaks: bool = False,
        annotate_break_kwargs: dict | None = None,
        filter_bands: tuple[float | None, float | None] = (0.1, 100.0),
        subset_chs: list | None = None,
        asr: bool | dict = False,
        filter_bands_ica: tuple[float | None, float | None] = (1.75, None),
        downsample_ica: float | None = 250.0,
        ica_method: str = "amica",
        amica_kwargs: dict | None = None,
        fit_ica: bool = True,
        thresh: float = -1,
        exclude_labels: list | None = None,
        include_labels: set | None = frozenset(
            {
                "brain",
                "muscle artifact",
                "heart beat",
                "line noise",
                "channel noise",
                "other",
            }
        ),
        fit_dipoles: bool = False,
        trans: object = None,
        rv_thresh: float | None = None,
        remove_outside_head: bool = False,
        rng_seed: int | None = None,
        event_id: dict | None = None,
        skip_if_exists: bool = False,
        make_report: bool = True,
        verbose: bool | str | int = True,
    ):
        if not fit_ica and fit_dipoles:
            raise ValueError(
                "Cannot fit dipoles without fitting ICA. "
                "Set fit_ica=True or fit_dipoles=False."
            )

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
        self.zapline_method = zapline_method
        self.get_bad_chs_kwargs = get_bad_chs_kwargs or {}
        self.annotate_breaks = annotate_breaks
        self.annotate_break_kwargs = annotate_break_kwargs
        self.filter_bands = filter_bands
        self.subset_chs = subset_chs
        self.asr = asr
        self.filter_bands_ica = filter_bands_ica
        self.downsample_ica = downsample_ica
        self.ica_method = ica_method
        self.amica_kwargs = amica_kwargs
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

        Parameters
        ----------
        fname_in : str | Path
            Path to the raw input file (XDF or any MNE-readable format).
        fname_out : str | Path | None
            Output stem for saving derivatives.  Pass ``None`` to skip saving.
        overwrite : bool
            Overwrite existing output files.

        Returns
        -------
        Same as :meth:`run_raw`.
        """
        rec = self.loader.load(fname_in)
        raw = rec.raw
        if self.channel_types:
            raw.set_channel_types(self.channel_types)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
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

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Recording to preprocess.
        fname_out : str | Path | None
            Output stem for saving derivatives.  Pass ``None`` to skip saving.
        overwrite : bool
            Overwrite existing output files.
        tier2 : dict | None
            Tier-2 streams from :class:`~bemobil_mne.io.MultimodalRecording`
            (e.g. ``rec.tier2``), kept at native rate and not merged into
            *raw*.  When provided and ``make_report=True``, each stream is
            plotted in full (decimated envelope, with drop-outs shaded) in
            its own report section.  ``None`` (default) skips this section.

        Returns
        -------
        raw_clean : mne.io.Raw
            ASR + ICA cleaned recording with bad channels interpolated.
        report : mne.Report | None
            Quality report (populated when ``make_report=True``, else ``None``).
        metadata : dict
            All other pipeline outputs keyed by name:
            ``raw_minimal``, ``raw_asr``, ``raw_subset``, ``ica``,
            ``ic_labels``, ``dipoles``, ``residuals``, ``trans``,
            ``bad_ch_dict``.
        """
        old_verbose = mne.set_log_level(verbose=self.verbose, return_old_level=True)
        timer = StepTimer()

        # --- Skip-if-exists caching ---
        if self.skip_if_exists and not overwrite and fname_out is not None:
            if Path(fname_out).exists():
                import logging as _log

                _log.getLogger(__name__).info(
                    f"skip_if_exists=True: loading cached outputs from {fname_out}"
                )
                return self._load_cached_outputs(fname_out)

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

        # --- ZapLine spectral cleaning ---
        if self.zapline_method is not None:
            t0 = time.perf_counter()
            zapline_freqs = _expand_line_noise_freq(
                self.line_noise_freq, raw.info["sfreq"]
            )
            raw = compute_zapline(
                raw, noise_freqs=zapline_freqs, method=self.zapline_method
            )
            append_desc(
                raw,
                name="zapline",
                method=self.zapline_method,
                noise_freqs=zapline_freqs.tolist(),
            )
            timer.log_step("zapline", time.perf_counter() - t0)

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

        # --- Minimal copy: bandpass + avg-ref projection ---
        t0 = time.perf_counter()
        raw_minimal = raw.copy()
        raw_minimal.filter(l_freq=self.filter_bands[0], h_freq=None)
        append_desc(
            raw_minimal,
            name="highpass",
            **sig_params(mne.io.Raw.filter, l_freq=self.filter_bands[0], h_freq=None),
        )
        _h_freq = self.filter_bands[1]
        if _h_freq is not None:
            _nyquist = raw_minimal.info["sfreq"] / 2
            if _h_freq >= _nyquist:
                _h_freq = _nyquist * 0.99
                warnings.warn(
                    "filter_bands h_freq clipped to "
                    f"{_h_freq:.2f} Hz (Nyquist = {_nyquist:.1f} Hz)",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raw_minimal.filter(l_freq=None, h_freq=_h_freq)
        append_desc(
            raw_minimal,
            name="lowpass",
            **sig_params(mne.io.Raw.filter, l_freq=None, h_freq=_h_freq),
        )
        raw_minimal.set_eeg_reference(ref_channels="average", projection=True)
        append_desc(raw_minimal, name="avg_ref_projection")
        timer.log_step("minimal_processing", time.perf_counter() - t0)

        # --- Channel subset (from minimal, no avg-ref applied) ---
        raw_subset = get_raw_subset(raw_minimal, subset_chs=self.subset_chs)
        if raw_subset is not None:
            raw_subset.del_proj()
            append_desc(
                raw_subset,
                name="subset_selection",
                channels=raw_subset.ch_names,
            )

        # --- ASR ---
        t0 = time.perf_counter()
        if self.asr is False:
            raw_asr = raw_minimal.copy()
        else:
            _asr_kwargs = self.asr if isinstance(self.asr, dict) else {}
            raw_asr = compute_asr(raw_minimal, **_asr_kwargs)
            append_desc(raw_asr, name="asr", **_asr_kwargs)
        timer.log_step("asr", time.perf_counter() - t0)

        # --- ICA ---
        dipoles, residuals = [], []
        trans_out = None
        # Snapshot annotations present at ICA time (after break annotation +
        # bad-channel detection, before ICA filtering/subsampling removes them)
        ica_annots = raw_asr.annotations.copy()
        t0 = time.perf_counter()

        if self.fit_ica:
            _ica_notch_freqs = _expand_line_noise_freq(
                self.line_noise_freq, raw_asr.info["sfreq"]
            )
            ica, ic_labels = compute_ica(
                raw_asr,
                filter_bands_ica=self.filter_bands_ica,
                notch_freqs=_ica_notch_freqs,
                downsample_ica=self.downsample_ica,
                thresh=self.thresh,
                rng_seed=self.rng_seed,
                exclude_labels=self.exclude_labels,
                include_labels=self.include_labels,
                ica_method=self.ica_method,
                amica_kwargs=self.amica_kwargs,
            )
            append_desc(
                raw_asr,
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
                fit_params=dict(ortho=False, extended=True),
            )  # stub - never fitted
            ic_labels = {}

        timer.log_step("ica", time.perf_counter() - t0)

        # --- Dipole fitting ---
        if self.fit_dipoles:
            t0 = time.perf_counter()
            trans_out = _handle_trans(self.trans, raw_asr.info)
            dipoles, residuals = fit_dipoles_on_ica(
                ica,
                raw_asr.info,
                trans_out,
                rv_thresh=self.rv_thresh,
                remove_outside_head=self.remove_outside_head,
            )
            n_valid = sum(d is not None for d in dipoles)
            append_desc(
                raw_asr,
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
            raw_clean = ica.apply(raw_asr.copy())
        else:
            raw_clean = raw_asr.copy()
        # Bake in the average EEG reference here, at the very end of the
        # pipeline. Up to this point the reference has only ever existed as
        # an unapplied SSP projection (added with projection=True back in
        # the "minimal" processing stage and inherited via .copy()/ICA
        # through raw_asr -> raw_clean and raw_minimal) -- it is only
        # actually applied to the data now, once, via apply_proj(). The
        # `set_eeg_reference` calls are only a defensive fallback in case a
        # given raw somehow doesn't already carry the projection (e.g. if
        # this method is ever called on a raw prepared outside the normal
        # run()/run_raw() flow).
        for _raw_ref in (raw_clean, raw_minimal):
            if not any(
                p["desc"] == "Average EEG reference" for p in _raw_ref.info["projs"]
            ):
                _raw_ref.set_eeg_reference(ref_channels="average", projection=True)
            _raw_ref.apply_proj()
        append_desc(raw_clean, name="avg_ref")
        append_desc(raw_minimal, name="avg_ref")
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

        # --- Report (optional) ---
        report = None
        if self.make_report:
            from bemobil_mne.preproc.make_report import make_report as _make_report

            report = _make_report(
                raw_minimal=raw_minimal,
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
            )

        # --- Save (optional) ---
        if fname_out is not None:
            t0 = time.perf_counter()
            self._save_outputs(
                fname_out,
                raw_minimal=raw_minimal,
                raw_clean=raw_clean,
                raw_asr=raw_asr,
                raw_subset=raw_subset,
                ica=ica,
                ic_labels=ic_labels,
                dipoles=dipoles,
                residuals=residuals,
                trans=trans_out,
                bad_ch_dict=bad_ch_dict,
                report=report,
                overwrite=overwrite,
            )
            timer.log_step("save", time.perf_counter() - t0)

        import logging as _logging

        _logging.getLogger(__name__).info(timer.format_summary())

        mne.set_log_level(verbose=old_verbose)

        metadata = {
            "raw_minimal": raw_minimal,
            "raw_asr": raw_asr,
            "raw_subset": raw_subset,
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
        report,
        overwrite,
    ):
        """Save all pipeline derivatives to disk."""
        fname_out = Path(fname_out)
        if fname_out.suffix == ".gz" and fname_out.stem.endswith(".fif"):
            stem = fname_out.with_name(fname_out.stem[:-4])  # strip .fif.gz
        else:
            stem = fname_out.with_suffix("")  # strip .fif
        fname_out.parent.mkdir(parents=True, exist_ok=True)

        # raw_clean is saved to fname_out directly (primary output / skip sentinel)
        raw_clean.save(fname_out, overwrite=overwrite, verbose="error")

        raw_minimal.save(
            stem.with_name(stem.name + "_minimal.fif.gz"),
            overwrite=overwrite,
            verbose="error",
        )
        if self.asr is not False:
            raw_asr.save(
                stem.with_name(stem.name + "_asr.fif.gz"),
                overwrite=overwrite,
                verbose="error",
            )
        if raw_subset is not None:
            raw_subset.save(
                stem.with_name(stem.name + "_minimal-subset.fif.gz"),
                overwrite=overwrite,
                verbose="error",
            )
        if self.fit_ica and ica.current_fit != "unfitted":
            ica.save(
                stem.with_name(stem.name + "_ica.fif.gz"),
                overwrite=overwrite,
            )
            with open(stem.with_name(stem.name + "_iclabels.json"), "w") as f:
                json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)

        with open(stem.with_name(stem.name + "_bad_channels.json"), "w") as f:
            json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)

        if dipoles:
            dipdir = stem.parent / f"{stem.name}_dipoles"
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
            trans.save(
                stem.with_name(stem.name + "_trans.fif"),
                overwrite=overwrite,
                verbose="error",
            )

        if report is not None:
            report_path = stem.with_name(stem.name + "_report.html")
            report.save(str(report_path), overwrite=overwrite, open_browser=False)

    def _load_cached_outputs(self, fname_out):
        """Load previously saved pipeline outputs and return the run_raw 3-tuple.

        Used by the ``skip_if_exists`` fast-path in :meth:`run_raw`.
        Missing auxiliary files yield ``None`` / empty collections.
        """
        fname_out = Path(fname_out)
        stem = fname_out.with_suffix("")

        raw_clean = mne.io.read_raw(fname_out, verbose="error")

        _minimal_path = stem.with_name(stem.name + "_minimal.fif.gz")
        raw_minimal = (
            mne.io.read_raw(_minimal_path, verbose="error")
            if _minimal_path.exists()
            else raw_clean
        )

        _asr_path = stem.with_name(stem.name + "_asr.fif.gz")
        raw_asr = (
            mne.io.read_raw(_asr_path, verbose="error")
            if _asr_path.exists()
            else raw_minimal
        )

        _subset_path = stem.with_name(stem.name + "_minimal-subset.fif.gz")
        raw_subset = (
            mne.io.read_raw(_subset_path, verbose="error")
            if _subset_path.exists()
            else None
        )

        _ica_path = stem.with_name(stem.name + "_ica.fif.gz")
        ica = (
            mne.preprocessing.read_ica(_ica_path, verbose="error")
            if _ica_path.exists()
            else mne.preprocessing.ICA(method="picard")
        )

        _labels_path = stem.with_name(stem.name + "_iclabels.json")
        ic_labels: dict = {}
        if _labels_path.exists():
            with open(_labels_path) as f:
                ic_labels = json.load(f)

        _bad_path = stem.with_name(stem.name + "_bad_channels.json")
        bad_ch_dict: dict = {}
        if _bad_path.exists():
            with open(_bad_path) as f:
                bad_ch_dict = json.load(f)

        _dipdir = stem.parent / f"{stem.name}_dipoles"
        dipoles: list = []
        residuals: list = []
        if _dipdir.exists():
            for dip_path in sorted(_dipdir.glob("ic-*-dip.bdip")):
                dipoles.append(mne.read_dipole(dip_path))
            for res_path in sorted(_dipdir.glob("ic-*-residual.fif.gz")):
                residuals.append(mne.read_evokeds(res_path, verbose="error")[0])

        _trans_path = stem.with_name(stem.name + "_trans.fif")
        trans = (
            mne.transforms.read_trans(_trans_path, verbose="error")
            if _trans_path.exists()
            else None
        )

        metadata = {
            "raw_minimal": raw_minimal,
            "raw_asr": raw_asr,
            "raw_subset": raw_subset,
            "ica": ica,
            "ic_labels": ic_labels,
            "dipoles": dipoles,
            "residuals": residuals,
            "trans": trans,
            "bad_ch_dict": bad_ch_dict,
        }
        return (raw_clean, None, metadata)
