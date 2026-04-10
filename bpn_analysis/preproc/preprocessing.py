"""EEG preprocessing pipeline."""

# %% Imports

from __future__ import annotations

import json
from pathlib import Path

import mne
import mne_faster
import mne_icalabel
import numpy as np
from meegkit.asr import ASR
from pyprep import NoisyChannels

# %% Functions


def get_bad_chs(
    raw,
    pyprep_kwargs=None,
    notch_lines=np.arange(50, 151, 50),
    notch_width=1.0,
):
    """Detect bad EEG channels using PyPREP and FASTER.

    Applies optional notch filtering and an average reference, then runs
    PyPREP's NoisyChannels (nan/flat & correlation methods) and FASTER's
    channel-level correlation on 1s fixed-length epochs. Returns a dictionary
    containing the union of identified bad channels and the per-method results.

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
    notch_lines : float | list of float | None
        The line frequencies for notch filtering. If ``None`` (default), do not notch
        filter.
    notch_width : float
        Width of the notch filter in Hz.

    Returns
    -------
    bad_ch_dict : dict
        Dictionary with the following keys:
        - "all_bads": list of unique bad channel names (union of PyPREP, FASTER,
          and manual bads).
        - "pyprep": dict returned by PyPREP's ``get_bads(as_dict=True)``.
        - "faster": dict returned by ``mne_faster.find_bad_channels(...)`` with
          per-metric lists and an entry "bad_all" containing the union of FASTER
          bads.
        - "bad_by_manual": list of manual bad channels (from ``pyprep_kwargs`` or
          ``raw.info['bads']``).
    """
    raw = raw.copy()

    # Consensus preprocessing steps before finding bads
    raw = raw.set_eeg_reference("average", projection=True)
    if notch_lines is not None:
        # potentially prune notch_lines for nyquist
        nyquist = raw.info["sfreq"] / 2
        notch_lines = notch_lines[: np.searchsorted(notch_lines, nyquist, side="right")]
        raw.notch_filter(freqs=notch_lines, notch_widths=notch_width)

    # === PyPREP ====
    # set default kwargs
    default_pyprep_kwargs = {
        "reject_by_annotation": "omit",
    }
    if pyprep_kwargs is not None:
        default_pyprep_kwargs.update(pyprep_kwargs)
    else:
        pyprep_kwargs = default_pyprep_kwargs

    bad_by_manual = pyprep_kwargs.get("bad_by_manual", [])  # get manual bads
    bad_by_manual = list(set(bad_by_manual + raw.info["bads"]))
    pyprep_kwargs.update({"bad_by_manual": bad_by_manual})

    # find bads using PyPREP's flat and correlation methods
    noisy_channels = NoisyChannels(raw, **pyprep_kwargs)
    noisy_channels.find_bad_by_nan_flat()
    noisy_channels.find_bad_by_correlation()
    bads_dict_pyprep = noisy_channels.get_bads(as_dict=True)

    # === FASTER ====
    # find bads using FASTER's correlation method
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

    # Get union of all bads
    bad_chs = set()
    for bads_dict in [bads_dict_pyprep, bads_dict_faster]:
        for bad_chs_list in bads_dict.values():
            bad_chs.update(bad_chs_list)
    bad_chs.update(bad_by_manual)

    bad_ch_dict = {
        "all_bads": list(bad_chs),
        "pyprep": bads_dict_pyprep,
        "faster": bads_dict_faster,
        "bad_by_manual": bad_by_manual,
    }

    # remove average reference projection
    raw.del_proj()

    return bad_ch_dict


# %% CLasses


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and float32."""

    def default(self, obj):
        """Convert numpy arrays and floats to JSON-serialisable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


class EEGPreprocessor:
    """Minimal EEG preprocessing pipeline: filter → bad channels → ICA → ASR → save.

    Parameters
    ----------
    loader : XDFLoader
        Configured loader used to read raw files.
    filter_bands : tuple of float
        (l_freq, h_freq) for the main bandpass filter applied to ``raw_minimal``.
    filter_bands_ica : tuple of float
        (l_freq, h_freq) for the ICA-specific bandpass filter.
    notch_freqs : array-like
        Line-noise frequencies to notch out (applied only on the ICA copy).
    downsample_ica : float
        Target sampling rate for ICA fitting.
    thresh : float
        ICLabel probability threshold above which a non-brain component is excluded.
    asr_cutoff : float
        ASR cutoff parameter (standard deviations above clean baseline).
        Lower values are more aggressive.  Typical range 5-20.
    rng_seed : int | None
        Random seed passed to ICA and PyPREP for reproducibility.
    """

    def __init__(
        self,
        loader,
        *,
        filter_bands: tuple[float | None, float | None] = (0.1, 100.0),
        filter_bands_ica: tuple[float | None, float | None] = (1.0, 100.0),
        notch_freqs: tuple[float, ...] = (50, 100, 150),
        downsample_ica: float = 250.0,
        thresh: float = 0.7,
        asr_cutoff: float = 20.0,
        rng_seed: int | None = None,
    ):
        self.loader = loader
        self.filter_bands = filter_bands
        self.filter_bands_ica = filter_bands_ica
        self.notch_freqs = notch_freqs
        self.downsample_ica = downsample_ica
        self.thresh = thresh
        self.asr_cutoff = asr_cutoff
        self.rng_seed = rng_seed

    def run(
        self,
        fname_in: str | Path,
        fname_out: str | Path,
        *,
        overwrite: bool = False,
    ) -> tuple[
        mne.io.BaseRaw,
        mne.io.BaseRaw,
        mne.io.BaseRaw,
        mne.preprocessing.ICA,
        dict,
        dict,
    ]:
        """Run the full preprocessing pipeline on a single file.

        Parameters
        ----------
        fname_in : str | Path
            Path to the raw input file (XDF or any MNE-readable format).
        fname_out : str | Path
            Output stem (extensions are added automatically).
            E.g. ``/data/sub-01/sub-01_preproc``.
        overwrite : bool
            Overwrite existing output files.

        Returns
        -------
        raw_minimal : mne.io.Raw
            Bandpass-filtered recording with average-reference projection.
        raw_clean : mne.io.Raw
            ICA-cleaned recording with bad channels interpolated.
        raw_asr : mne.io.Raw
            ASR-cleaned copy of ``raw_clean``.
        ica : mne.preprocessing.ICA
            Fitted ICA object with ``exclude`` set.
        ic_labels : dict
            ICLabel classification results.
        bad_ch_dict : dict
            Bad channel detection results from PyPREP / FASTER.
        """
        fname_out = Path(fname_out).with_suffix("")
        raw = self.loader.load(fname_in)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

        # --- Bad channel detection ---
        bad_ch_dict = get_bad_chs(
            raw,
            pyprep_kwargs={"random_state": self.rng_seed},
            notch_lines=self.notch_freqs,
            notch_width=1.0,
        )
        raw.info["bads"] = bad_ch_dict["all_bads"]

        # --- Minimal copy: bandpass + avg-ref projection ---
        raw_minimal = raw.copy()
        raw_minimal.filter(l_freq=self.filter_bands[0], h_freq=None)
        raw_minimal.filter(l_freq=None, h_freq=self.filter_bands[1])
        raw_minimal.set_eeg_reference(ref_channels="average", projection=True)

        # --- ICA copy: stricter HP, notch, downsample, avg-ref ---
        raw_ica = raw.copy().pick("eeg")
        raw_ica.filter(l_freq=self.filter_bands_ica[0], h_freq=None)
        raw_ica.filter(l_freq=None, h_freq=self.filter_bands_ica[1])
        raw_ica.notch_filter(freqs=self.notch_freqs, notch_widths=1.0)
        if raw_ica.info["sfreq"] > self.downsample_ica:
            raw_ica.resample(self.downsample_ica)
        raw_ica.set_eeg_reference(ref_channels="average")

        # Fixed-length epochs; reject annotated bad segments
        epochs = mne.make_fixed_length_epochs(
            raw_ica, duration=1.0, preload=True, reject_by_annotation=True
        )

        # Drop noisy epochs via FASTER
        bad_epochs = mne_faster.find_bad_epochs(epochs)
        if len(bad_epochs) > 0:
            epochs.drop(bad_epochs)

        # --- ICA ---
        ica = mne.preprocessing.ICA(
            n_components=None,
            random_state=self.rng_seed,
            method="picard",
            fit_params=dict(ortho=False, extended=True),
        )
        ica.fit(epochs)

        ic_labels = mne_icalabel.label_components(epochs, ica, method="iclabel")

        keep_labels = {"brain", "other"}
        ica.exclude = [
            idx
            for idx, (label, prob) in enumerate(
                zip(ic_labels["labels"], ic_labels["y_pred_proba"])
            )
            if label not in keep_labels and prob >= self.thresh
        ]

        # --- Apply ICA, avg-ref, interpolate ---
        raw_clean = ica.apply(raw_minimal.copy())
        raw_clean.set_eeg_reference(ref_channels="average")
        raw_minimal.set_eeg_reference(ref_channels="average")
        raw_clean.interpolate_bads(reset_bads=True, method="spline")

        # --- ASR ---
        asr = ASR(sfreq=raw_clean.info["sfreq"], cutoff=self.asr_cutoff)
        eeg_idx = mne.pick_types(raw_clean.info, eeg=True)
        eeg_data = raw_clean.get_data(picks="eeg")
        asr.fit(eeg_data)
        eeg_clean_asr = asr.transform(eeg_data)
        raw_asr = raw_clean.copy()
        raw_asr._data[eeg_idx] = eeg_clean_asr

        # --- Save ---
        fname_out.parent.mkdir(parents=True, exist_ok=True)
        raw_minimal.save(
            fname_out.with_name(fname_out.name + "_minimal.fif.gz"), overwrite=overwrite
        )
        raw_clean.save(
            fname_out.with_name(fname_out.name + "_clean.fif.gz"), overwrite=overwrite
        )
        raw_asr.save(
            fname_out.with_name(fname_out.name + "_asr.fif.gz"), overwrite=overwrite
        )
        ica.save(
            fname_out.with_name(fname_out.name + "_ica.fif.gz"), overwrite=overwrite
        )

        with open(fname_out.with_name(fname_out.name + "_bad_channels.json"), "w") as f:
            json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)
        with open(fname_out.with_name(fname_out.name + "_iclabels.json"), "w") as f:
            json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)

        return raw_minimal, raw_clean, raw_asr, ica, ic_labels, bad_ch_dict
