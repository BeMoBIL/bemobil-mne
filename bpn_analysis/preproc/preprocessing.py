"""EEG preprocessing pipeline."""

# %% Imports

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import mne
import mne_faster
import numpy as np
from pyprep import NoisyChannels

from bpn_analysis.io.utils import NumpyEncoder as _NumpyEncoder
from bpn_analysis.preproc.utils import (
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

# %% Functions


def get_bad_chs(
    raw,
    pyprep_kwargs=None,
    notch_lines=np.arange(50, 151, 50),
    notch_width=1.0,
    line_noise_crit=4.0,
    flatline_crit=5.0,
):
    """Detect bad EEG channels using PyPREP, FASTER, flatline, and line noise.

    Applies optional notch filtering and an average reference, then runs
    PyPREP's NoisyChannels (nan/flat & correlation methods), FASTER's
    channel-level correlation on 1s fixed-length epochs, per-channel flatline
    detection, and per-channel line-noise z-score detection.  Returns a
    dictionary containing the union of all identified bad channels and
    per-method breakdowns.

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
        Pass ``None`` to skip this criterion.
    flatline_crit : float | None
        Maximum allowed duration (seconds) of a flatline segment.  Channels
        that are flat for longer than this threshold are flagged as bad via
        PyPREP's ``find_bad_by_nan_flat``.  Pass ``None`` to disable.

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
    noisy_channels.find_bad_by_correlation()
    bads_dict_pyprep = noisy_channels.get_bads(as_dict=True)

    # === FASTER ====
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True)
    picks = mne.pick_types(epochs.info, eeg=True, exclude=bad_by_manual)
    epochs.pick(picks)
    bads_dict_faster = mne_faster.find_bad_channels(
        epochs,
        return_by_metric=True,
        use_metrics=["correlation"],
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


class EEGPreprocessor:
    """EEG preprocessing pipeline: filter → bad channels → ASR → ICA → save.

    Mirrors the ``run_preprocessing`` API from ``standard_scripts`` while
    following the ``bpn_analysis`` code conventions.

    Parameters
    ----------
    loader : XDFLoader | None
        Configured loader used by :meth:`run` to read raw files.
        Not required when calling :meth:`run_raw` directly.
    filter_bands : tuple of float
        ``(l_freq, h_freq)`` for the main bandpass filter applied to
        ``raw_minimal``.
    filter_bands_ica : tuple of float
        ``(l_freq, h_freq)`` for the ICA-specific bandpass filter.
    notch_freqs : array-like | ``"europe"`` | ``"usa"``
        Line-noise frequencies to notch out. Strings ``"europe"`` and
        ``"usa"`` expand to 50/100/150 Hz and 60/120/180 Hz respectively.
    downsample_ica : float | None
        Target sampling rate for ICA fitting.  ``None`` skips downsampling.
    thresh : float
        ICLabel probability threshold above which a component is excluded.
    asr_cutoff : float | False
        ASR cutoff parameter.  Pass ``False`` to skip ASR entirely.
    rng_seed : int | None
        Random seed for ICA and PyPREP.
    annotate_break_kwargs : dict | None
        Forwarded to :func:`mne.preprocessing.annotate_break`.
    exclude_labels : list of str | None
        ICLabel categories to exclude.  Mutually exclusive with
        *include_labels*.
    include_labels : set of str | None
        ICLabel categories to keep; all others are excluded.
        Defaults to ``{"brain", "other"}``.  Mutually exclusive with
        *exclude_labels*.
    channel_types : dict | None
        Channel name → MNE type mapping applied right after loading.
    ica_method : str
        ICA algorithm.  ``"amica"`` (default) uses AMICA via
        ``amica-python`` and converts to MNE ICA; falls back to picard if the
        package is not installed.  Any other string is forwarded as the
        ``method`` argument to :class:`mne.preprocessing.ICA` (e.g.
        ``"picard"``, ``"fastica"``).  Ignored when ``fit_ica=False``.
    fit_ica : bool
        If ``False``, skip ICA entirely (``raw_clean`` equals ``raw_asr``).
    fit_dipoles : bool
        If ``True``, fit a dipole to each ICA component topography using the
        fsaverage BEM.  Requires a montage with digitisation.
    trans : mne.transforms.Transform | ``"fit"`` | ``"fsaverage"`` | None
        Head→MRI transform for dipole fitting. ``None`` and ``"fsaverage"``
        use the MNE built-in template; ``"fit"`` runs automatic coregistration.
        Ignored when ``fit_dipoles=False``.
    subset_chs : list of str | None
        Channels for the ``raw_subset`` output (minimally processed, without
        average reference).  Defaults to the some central and frontal channels when
        ``None`` but produces ``None`` if none are found in the data.
    pre_hook : callable | None
        Called on the raw object before any processing.  Must accept a raw
        object and return either a raw object or a ``(raw, description)`` tuple
        where *description* is a string recorded in the provenance.
    verbose : bool | str | int
        MNE verbosity level during processing.
    event_id : dict | None
        Event map recorded in provenance metadata (no effect on processing).
    pyprep_kwargs : dict | None
        Extra keyword arguments forwarded to PyPREP's
        :class:`~pyprep.NoisyChannels`.  The ``random_state`` key is
        always overwritten with *rng_seed*.
    line_noise_crit : float | None
        Z-score threshold for the per-channel line-noise criterion in bad
        channel detection.  ``None`` disables this criterion.
    zapline_freqs : array-like | ``"europe"`` | ``"usa"`` | None
        If not ``None``, applies ZapLine (DSS-based) spectral cleaning to the
        main raw recording before the bandpass filter.  Accepts the same
        string shortcuts as *notch_freqs*.  Pass ``None`` together with
        ``zapline_method="adaptive"`` to enable auto-detection.
    zapline_method : str
        Algorithm used by :func:`compute_zapline`.  One of:

        ``"adaptive"`` (default)
            ZapLine-plus (mne-denoise) with adaptive frequency detection.
        ``"zapline"``
            Standard ZapLine (mne-denoise), fixed-frequency.
        ``"dss_line"``
            Single-pass DSS (meegkit).
        ``"dss_line_iter"``
            Iterative DSS (meegkit).
    rename_channels : dict | str | None
        Channel renaming applied immediately before any processing.

        - ``dict``: explicit ``{old_name: new_name}`` mapping.
        - ``str``: strip this prefix from every channel name that starts
          with it (e.g. ``"BrainVision RDA_"``).
        - ``None`` (default): no renaming.
    final_filter_bands : tuple of float | None
        If not ``None``, apply a final bandpass filter (MNE ``raw.filter``)
        to ``raw_clean`` after ICA cleaning, average re-referencing, and bad
        channel interpolation.  Useful for removing very slow drifts before
        further analysis (e.g. ``(0.5, 40.0)``).
    rv_thresh : float | None
        Residual-variance threshold for dipole fitting.  Components whose
        best-fitting dipole has RV >= *rv_thresh* are set to ``None`` in the
        ``dipoles`` and ``residuals`` output lists.  ``None`` keeps all
        dipoles.  Typical value: ``0.15`` (15 %).
    remove_outside_head : bool
        If ``True``, components whose dipole falls outside the head model
        (position norm > 0.13 m) are set to ``None`` in the output lists.
    skip_if_exists : bool
        If ``True`` *and* ``overwrite=False``, skip the entire computation
        when the primary output file ``{fname_out}_clean.fif.gz`` already
        exists and return the previously saved results instead.
    make_report : bool
        If ``True`` and *fname_out* is provided, generate an
        :class:`mne.Report` summarising the preprocessing outputs and save it
        alongside the other derivatives as ``{fname_out}_report.html``.
    """

    def __init__(
        self,
        loader,
        *,
        filter_bands: tuple[float | None, float | None] = (0.1, 100.0),
        filter_bands_ica: tuple[float | None, float | None] = (1.0, 100.0),
        notch_freqs: tuple[float, ...] | str = (50, 100, 150),
        downsample_ica: float | None = 250.0,
        thresh: float = 0.7,
        asr_cutoff: float | bool = 20.0,
        rng_seed: int | None = None,
        annotate_break_kwargs: dict | None = None,
        exclude_labels: list | None = None,
        include_labels: set | None = frozenset({"brain", "other"}),
        channel_types: dict | None = None,
        ica_method: str = "amica",
        fit_ica: bool = True,
        fit_dipoles: bool = False,
        trans: object = None,
        subset_chs: list | None = None,
        pre_hook: object = None,
        verbose: bool | str | int = True,
        event_id: dict | None = None,
        pyprep_kwargs: dict | None = None,
        line_noise_crit: float | None = 4.0,
        zapline_freqs=None,
        zapline_method: str = "adaptive",
        rename_channels=None,
        final_filter_bands: tuple[float | None, float | None] | None = None,
        rv_thresh: float | None = None,
        remove_outside_head: bool = False,
        skip_if_exists: bool = False,
        make_report: bool = False,
    ):
        if not fit_ica and fit_dipoles:
            raise ValueError(
                "Cannot fit dipoles without fitting ICA. "
                "Set fit_ica=True or fit_dipoles=False."
            )

        # Expand notch string shortcuts so every internal caller sees a tuple
        if isinstance(notch_freqs, str):
            if notch_freqs == "europe":
                notch_freqs = (50, 100, 150)
            elif notch_freqs == "usa":
                notch_freqs = (60, 120, 180)
            else:
                raise ValueError(
                    f"Unknown notch_freqs preset: {notch_freqs!r}. "
                    "Use 'europe', 'usa', or an explicit tuple."
                )

        self.loader = loader
        self.filter_bands = filter_bands
        self.filter_bands_ica = filter_bands_ica
        self.notch_freqs = notch_freqs
        self.downsample_ica = downsample_ica
        self.thresh = thresh
        self.asr_cutoff = asr_cutoff
        self.rng_seed = rng_seed
        self.annotate_break_kwargs = annotate_break_kwargs
        self.exclude_labels = exclude_labels
        self.include_labels = include_labels
        self.channel_types = channel_types
        self.ica_method = ica_method
        self.fit_ica = fit_ica
        self.fit_dipoles = fit_dipoles
        self.trans = trans
        self.subset_chs = subset_chs
        self.pre_hook = pre_hook
        self.verbose = verbose
        self.event_id = event_id
        self.pyprep_kwargs = pyprep_kwargs or {}
        self.line_noise_crit = line_noise_crit
        self.zapline_freqs = zapline_freqs
        self.zapline_method = zapline_method
        self.rename_channels = rename_channels
        self.final_filter_bands = final_filter_bands
        self.rv_thresh = rv_thresh
        self.remove_outside_head = remove_outside_head
        self.skip_if_exists = skip_if_exists
        self.make_report = make_report

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
        raw = self.loader.load(fname_in)
        if self.channel_types:
            raw.set_channel_types(self.channel_types)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
        return self.run_raw(raw, fname_out=fname_out, overwrite=overwrite)

    def run_raw(
        self,
        raw: mne.io.BaseRaw,
        fname_out: str | Path | None = None,
        *,
        overwrite: bool = False,
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
        if self.pre_hook is not None:
            if not callable(self.pre_hook):
                raise TypeError("`pre_hook` must be callable.")
            result = self.pre_hook(raw)
            if isinstance(result, tuple):
                raw, pre_hook_description = result
            else:
                raw = result
            if pre_hook_description:
                append_desc(raw, name="pre_hook", description=pre_hook_description)

        # --- ZapLine spectral cleaning ---
        if self.zapline_freqs is not None:
            t0 = time.perf_counter()
            raw = compute_zapline(
                raw, noise_freqs=self.zapline_freqs, method=self.zapline_method
            )
            append_desc(
                raw,
                name="zapline",
                method=self.zapline_method,
                noise_freqs=list(
                    np.atleast_1d(self.zapline_freqs).tolist()
                    if not isinstance(self.zapline_freqs, str)
                    else self.zapline_freqs
                ),
            )
            timer.log_step("zapline", time.perf_counter() - t0)

        # --- Bad channel detection ---
        t0 = time.perf_counter()
        pyprep_kw = dict(self.pyprep_kwargs)
        pyprep_kw["random_state"] = self.rng_seed
        bad_ch_dict = get_bad_chs(
            raw,
            pyprep_kwargs=pyprep_kw,
            notch_lines=np.asarray(self.notch_freqs),
            notch_width=1.0,
            line_noise_crit=self.line_noise_crit,
        )
        raw.info["bads"] = bad_ch_dict["all_bads"]
        append_desc(
            raw,
            name="bad_channel_detection",
            all_bads=bad_ch_dict["all_bads"],
        )
        timer.log_step("bad_channel_detection", time.perf_counter() - t0)

        # --- Annotate breaks ---
        t0 = time.perf_counter()
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
        if self.asr_cutoff is False:
            raw_asr = raw_minimal.copy()
        else:
            raw_asr = compute_asr(raw_minimal, cutoff=self.asr_cutoff)
            append_desc(raw_asr, name="asr", cutoff=self.asr_cutoff)
        timer.log_step("asr", time.perf_counter() - t0)

        # --- ICA ---
        dipoles, residuals = [], []
        trans_out = None
        t0 = time.perf_counter()

        if self.fit_ica:
            ica, ic_labels = compute_ica(
                raw_asr,
                filter_bands_ica=self.filter_bands_ica,
                notch_freqs=self.notch_freqs,
                downsample_ica=self.downsample_ica,
                thresh=self.thresh,
                rng_seed=self.rng_seed,
                exclude_labels=self.exclude_labels,
                include_labels=self.include_labels,
                ica_method=self.ica_method,
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
        raw_clean.set_eeg_reference(ref_channels="average")
        append_desc(raw_clean, name="avg_ref")
        raw_minimal.set_eeg_reference(ref_channels="average")
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

        # --- Post-ICA final bandpass ---
        if self.final_filter_bands is not None:
            l_freq, h_freq = self.final_filter_bands
            if l_freq is not None:
                raw_clean.filter(l_freq=l_freq, h_freq=None)
            if h_freq is not None:
                raw_clean.filter(l_freq=None, h_freq=h_freq)
            append_desc(
                raw_clean,
                name="final_filter",
                l_freq=l_freq,
                h_freq=h_freq,
            )

        timer.log_step("rereference_interpolate", time.perf_counter() - t0)

        # --- Report (optional) ---
        report = None
        if self.make_report:
            from bpn_analysis.preproc.make_report import make_report as _make_report

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
        stem = fname_out.with_suffix("")
        fname_out.parent.mkdir(parents=True, exist_ok=True)

        # raw_clean is saved to fname_out directly (primary output / skip sentinel)
        raw_clean.save(fname_out, overwrite=overwrite, verbose="error")

        raw_minimal.save(
            stem.with_name(stem.name + "_minimal.fif.gz"),
            overwrite=overwrite,
            verbose="error",
        )
        if self.asr_cutoff is not False:
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
