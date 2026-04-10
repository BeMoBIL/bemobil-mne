"""Analyze HIP data for mne-denoise."""

# %% Imports

import gc
import json
import re
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import mne_faster
import mne_icalabel
import numpy as np
from dataqa import get_bad_chs
from meegkit.asr import ASR
from standard_scripts import plot_ERP

from bpn_analysis import XDFLoader

# %% Constants & Settings

DATA_DIR = Path(r"C:\Users\random\Documents\Data\young HIP")
DERIV_DIR = DATA_DIR / "derivatives"

# let's only use visual for now
REMAPS = {
    # "STaudio": {
    #     "ST_audio/left_high": "incongruent",
    #     "ST_audio/left_low": "congruent",
    #     "ST_audio/right_high": "congruent",
    #     "ST_audio/right_low": "incongruent",
    # },
    "STvisual": {
        "ST_visual/left_second": "incongruent",
        "ST_visual/left_first": "congruent",
        "ST_visual/right_second": "congruent",
        "ST_visual/right_first": "incongruent",
    },
    # "DTaudio": {
    #     "DT_audio/left_high": "incongruent",
    #     "DT_audio/left_low": "congruent",
    #     "DT_audio/right_high": "congruent",
    #     "DT_audio/right_low": "incongruent",
    # },
    "DTvisual": {
        "DT_visual/left_second": "incongruent",
        "DT_visual/left_first": "congruent",
        "DT_visual/right_second": "congruent",
        "DT_visual/right_first": "incongruent",
    },
}

EPOCH_TLIMES = (-0.2, 0.8)
BASELINE = (-0.2, 0)

BANDPASS_ERP = (None, 20.0)

# Stimulus onset delay: shift trigger annotations forward by this amount so
# that t=0 in the epoch aligns with actual stimulus presentation.
# 60 ms is a typical LSL/screen-refresh latency for this setup.
TSHIFT = 0.060  # seconds

FORCE_RERUN = False  # set True to reprocess subjects even if all outputs exist

LOADER = XDFLoader(
    eeg_stream_type="EEG",
    montage="standard_1005",
    drop_channels=["x_dir", "y_dir", "z_dir"],
    target_sfreq=500.0,
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


def copy_data(src_root=None, dst_root=None, young_subjects=None):
    """Copy data from source to destination, preserving structure."""
    if src_root is None:
        src_root = Path(
            r"\\stor1.bpn.tu-berlin.de\projects\Project_HearingImpaired\Recordings"
        )
    if dst_root is None:
        dst_root = Path(r"C:\Users\random\Documents\Data\young HIP")
    if young_subjects is None:
        young_subjects = [
            "p001",
            "p002",
            "p003",
            "p004",
            "p005",
            "p006",
            "p007",
            "p008",
            "p009",
            "p011",
            "p013-2",
            "p014",
            "p016",
            "p017",
            "p018",
            "p019",
            "p021",
            "p022",
            "p023",
            "p024",
            "p027",
            "p028",
            "p029",
            "p030",
            "p032",
            "p034",
            "p035",
            "p036",
            "p037",
            "p042",
            "p043",
            "p054",
            "p056",
            "p057",
            "p058",
            "p061",
            "p063",
            "p065",
            "p066",
            "p067",
            "p068",
            "p077",
            "p087",
            "p094",
        ]

    dst_root.mkdir(parents=True, exist_ok=True)

    for subject in young_subjects:
        src_dir = src_root / subject

        if not src_dir.exists():
            print(f"[SKIP]  {subject} — not found in source")
            continue

        xdf_files = [
            f
            for f in src_dir.rglob("*.xdf")
            if "old" not in f.name.lower() and "baseline" not in f.name.lower()
        ]

        if not xdf_files:
            print(f"[SKIP]  {subject} — no matching XDF files")
            continue

        for src_file in xdf_files:
            # Preserve subfolder structure within the subject folder
            relative = src_file.relative_to(src_root)
            dst_file = dst_root / relative

            if dst_file.exists():
                print(f"[SKIP]  {relative} — already exists")
                continue

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[COPY]  {relative} ...", end=" ", flush=True)
            shutil.copy2(src_file, dst_file)
            print("done")

    print("\nFinished.")


def clear_matplotlib_memory():
    """Clear and close all matplotlib figures and caches."""
    plt.close("all")
    matplotlib._pylab_helpers.Gcf.destroy_all()
    gc.collect()


def get_stimulus_rename_map(descriptions):
    """Get event remap."""
    unique_labels = set()

    for desc in descriptions:
        if not desc.startswith("trialStart"):
            continue
        condition_match = re.search(r"condition:(\w+)", desc)
        stimulus_match = re.search(r"stimulus:(\w+)", desc)
        if condition_match and stimulus_match:
            unique_labels.add(f"{condition_match.group(1)}/{stimulus_match.group(1)}")

    unique_labels = sorted(unique_labels)
    label_to_id = {label: i + 1 for i, label in enumerate(unique_labels)}

    rename_map = {}
    for desc in descriptions:
        if not desc.startswith("trialStart"):
            continue
        condition_match = re.search(r"condition:(\w+)", desc)
        stimulus_match = re.search(r"stimulus:(\w+)", desc)
        if condition_match and stimulus_match:
            label = f"{condition_match.group(1)}/{stimulus_match.group(1)}"
            rename_map[desc] = label

    print("Discovered event types:")
    for label, id_ in label_to_id.items():
        print(f"  {id_}: {label}")

    return rename_map, label_to_id


# %% Functions


def run_preprocessing(
    fname_in,
    fname_out,
    *,
    filter_bands=(0.1, 100.0),
    filter_bands_ica=(1.0, 100.0),
    notch_freqs=(50, 100, 150),
    downsample_ica=250,
    thresh=0.7,
    asr_cutoff=20,
    rng_seed=None,
    overwrite=False,
):
    """Minimal EEG preprocessing: filter → bad channels → ICA → clean → save.

    Parameters
    ----------
    fname_in : str | Path
        Path to a single XDF (or any MNE-readable) file.
    fname_out : str | Path
        Output stem, e.g. ``/data/sub-01/sub-01_preproc``.
        Extensions are added automatically.
    filter_bands : tuple of float
        (l_freq, h_freq) for the main bandpass filter.
    filter_bands_ica : tuple of float
        (l_freq, h_freq) for the ICA-specific bandpass filter.
    notch_freqs : array-like
        Line noise frequencies to notch out.
    downsample_ica : float
        Target sampling rate for ICA (anti-aliasing is applied automatically).
    thresh : float
        ICLabel probability threshold for artifact rejection.
    asr_cutoff : float
        ASR cutoff parameter (standard deviations above the clean baseline
        before a component is reconstructed).  Lower = more aggressive.
        Typical range 5–20; default 20 is conservative.
    rng_seed : int | None
        Random seed for ICA reproducibility.
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    raw_minimal : mne.io.Raw
    raw_clean   : mne.io.Raw
    raw_asr     : mne.io.Raw
    ica         : mne.preprocessing.ICA
    ic_labels   : dict
    bad_ch_dict : dict
    """
    from pathlib import Path

    fname_out = Path(fname_out).with_suffix("")
    raw = LOADER.load(fname_in)

    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

    # Bad channel detection (uses PyPREP + FASTER internals)
    bad_ch_dict = get_bad_chs(
        raw,
        pyprep_kwargs={"random_state": rng_seed},
        notch_lines=notch_freqs,
        notch_width=1.0,
    )
    raw.info["bads"] = bad_ch_dict["all_bads"]

    # Minimal processing: filter + avg-ref projection (not applied yet)
    raw_minimal = raw.copy()

    raw_minimal.filter(l_freq=filter_bands[0], h_freq=None)
    raw_minimal.filter(l_freq=None, h_freq=filter_bands[1])

    # raw_minimal.notch_filter(freqs=notch_freqs, notch_widths=1.0)
    raw_minimal.set_eeg_reference(ref_channels="average", projection=True)

    # Prepare ICA copy: stricter HP, downsample, avg-ref
    raw_ica = raw.copy().pick("eeg")

    raw_ica.filter(l_freq=filter_bands_ica[0], h_freq=None)
    raw_ica.filter(l_freq=None, h_freq=filter_bands_ica[1])

    raw_ica.notch_filter(freqs=notch_freqs, notch_widths=1.0)
    if raw_ica.info["sfreq"] > downsample_ica:
        raw_ica.resample(downsample_ica)
    raw_ica.set_eeg_reference(ref_channels="average")

    # Fixed-length epochs, reject annotated bads
    epochs = mne.make_fixed_length_epochs(
        raw_ica, duration=1.0, preload=True, reject_by_annotation=True
    )

    # Drop noisy epochs via FASTER
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

    keep_labels = {"brain", "other"}
    exclude_idx = [
        idx
        for idx, (label, prob) in enumerate(
            zip(ic_labels["labels"], ic_labels["y_pred_proba"])
        )
        if label not in keep_labels and prob >= thresh
    ]
    ica.exclude = exclude_idx

    # Apply ICA, avg-ref, interpolate bad channels
    raw_clean = ica.apply(raw_minimal.copy())
    raw_clean.set_eeg_reference(ref_channels="average")
    raw_minimal.set_eeg_reference(ref_channels="average")
    raw_clean.interpolate_bads(reset_bads=True, method="spline")

    # ASR — fit on ICA-cleaned data, apply to produce raw_asr
    # ASR calibrates on artifact-free windows (selected automatically) and
    # reconstructs artifact-contaminated subspaces in the full recording.
    # raw_asr is a copy of raw_clean with residual non-stationary artifacts
    # (e.g. movement bursts) further suppressed.
    asr = ASR(sfreq=raw_clean.info["sfreq"], cutoff=asr_cutoff)
    eeg_idx = mne.pick_types(raw_clean.info, eeg=True)
    eeg_data = raw_clean.get_data(picks="eeg")
    asr.fit(eeg_data)
    eeg_clean_asr = asr.transform(eeg_data)
    raw_asr = raw_clean.copy()
    raw_asr._data[eeg_idx] = eeg_clean_asr

    # Save
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
    ica.save(fname_out.with_name(fname_out.name + "_ica.fif.gz"), overwrite=overwrite)

    with open(fname_out.with_name(fname_out.name + "_bad_channels.json"), "w") as f:
        json.dump(bad_ch_dict, f, indent=4, cls=NumpyEncoder)
    with open(fname_out.with_name(fname_out.name + "_iclabels.json"), "w") as f:
        json.dump(ic_labels, f, indent=4, cls=NumpyEncoder)

    return raw_minimal, raw_clean, raw_asr, ica, ic_labels, bad_ch_dict


def run_preparation(raw_clean, fname_out, cond, tshift=TSHIFT, overwrite=False):
    """Run preparation steps: event remapping, filtering, epoching, etc.

    Parameters
    ----------
    tshift : float
        Stimulus onset delay in seconds.  Annotation onsets are shifted
        forward by this amount so that t=0 in each epoch aligns with actual
        stimulus presentation rather than trigger receipt.
        Set to 0 to disable.  Default: TSHIFT (80 ms).
    """
    rename_map, _ = get_stimulus_rename_map(raw_clean.annotations.description)
    remap = REMAPS[cond]

    raw_remap = raw_clean.copy()
    raw_remap.annotations.rename(rename_map)
    raw_remap.annotations.rename(remap)

    if tshift != 0:
        shifted = raw_remap.annotations.copy()
        shifted.onset += tshift
        raw_remap.set_annotations(shifted)

    raw_remap.filter(l_freq=BANDPASS_ERP[0], h_freq=BANDPASS_ERP[1])

    mask = [a in remap.values() for a in raw_remap.annotations.description]
    new_annots = raw_remap.annotations[mask]
    raw_remap.set_annotations(new_annots)

    events, ids = mne.events_from_annotations(raw_remap)
    id_of_interest = {ev: ids[ev] for ev in remap.values()}

    epochs = mne.Epochs(
        raw_remap,
        event_id=id_of_interest,
        events=events,
        tmin=EPOCH_TLIMES[0],
        tmax=EPOCH_TLIMES[1],
        baseline=BASELINE,
    )

    epochs.save(
        fname_out.with_name(fname_out.name + "_epo.fif.gz"), overwrite=overwrite
    )

    epochs_dict = {}
    for label in remap.values():
        epochs_dict[label] = epochs[label]

    return epochs_dict


def main():
    """Run main function."""
    for cond in REMAPS.keys():
        files = list(DATA_DIR.rglob(f"*{cond}.xdf"))

        for fpath in files:
            fpath_out = DERIV_DIR / fpath.parent.name / f"{fpath.stem}_preproc"

            expected_outputs = [
                fpath_out.with_name(fpath_out.name + "_minimal.fif.gz"),
                fpath_out.with_name(fpath_out.name + "_clean.fif.gz"),
                fpath_out.with_name(fpath_out.name + "_asr.fif.gz"),
                fpath_out.with_name(fpath_out.name + "_ica.fif.gz"),
                fpath_out.with_name(fpath_out.name + "_bad_channels.json"),
                fpath_out.with_name(fpath_out.name + "_iclabels.json"),
                fpath_out.with_name(f"{fpath.stem}_ERP.png"),
            ]
            if not FORCE_RERUN and all(p.exists() for p in expected_outputs):
                print(f"[SKIP] {fpath.stem} — all outputs present")
                continue

            # get data preprocessed
            raw_minimal, raw_clean, raw_asr, ica, ic_labels, bad_ch_dict = (
                run_preprocessing(
                    fpath,
                    fpath_out,
                    overwrite=True,
                )
            )

            # get data sorted out
            epochs_dict = run_preparation(raw_clean, fpath_out, cond, overwrite=True)

            # single-subject plot
            fig = plot_ERP(epochs_dict)
            fig.savefig(fpath_out.with_name(f"{fpath.stem}_ERP.png"), dpi=300)
            clear_matplotlib_memory()


# %% Main
if __name__ == "__main__":
    # copy_data()
    main()

# %%
