"""BIDS export using mne_bids wrappers in bemobil_mne.io.

Demonstrates:
- export_to_bids: export a single Raw object to a BIDS directory tree.
- make_bids_dataset_description: write dataset_description.json.
- batch_export_to_bids: export multiple subjects/runs in one call.

Source data: MNE eegbci motor-imagery dataset (downloaded automatically).
The BIDS output is written to a temporary directory that is cleaned up at
the end of the script.
"""

# %% Imports

import tempfile
from pathlib import Path

import mne
from mne.datasets import eegbci

from bemobil_mne.io import (
    batch_export_to_bids,
    export_to_bids,
    make_bids_dataset_description,
)

# %% Settings & Constants

# eegbci run 6 = motor imagery (left/right hand), subject 1
SUBJECT_ID = 1
RUNS = [6, 10]  # two runs from the same subject
LINE_FREQ = 60.0  # PhysioNet data recorded in the USA

# %% Load source data (MNE eegbci)

raws = []
for run in RUNS:
    fnames = eegbci.load_data(subject=SUBJECT_ID, runs=[run])
    raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
    eegbci.standardize(raw)  # rename channels to 10-20 labels
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, verbose=False)
    raws.append(raw)

print(f"Loaded {len(raws)} runs, {raws[0].info['nchan']} channels each")

# %% Single-run export

with tempfile.TemporaryDirectory() as tmpdir:
    bids_root = Path(tmpdir) / "bids_single"

    # Write dataset_description.json first
    make_bids_dataset_description(
        bids_root=bids_root,
        name="eegbci-motor-imagery-demo",
        authors=["BPN Lab"],
        data_license="CC0",
    )

    bids_path = export_to_bids(
        raw=raws[0],
        bids_root=bids_root,
        subject="01",
        session="01",
        task="motorImagery",
        run="06",
        line_freq=LINE_FREQ,
        overwrite=True,
        verbose=False,
    )

    print("Single export written to:", bids_path.fpath)
    print("BIDS tree:")
    for p in sorted(bids_root.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(bids_root))

# %% Batch export

with tempfile.TemporaryDirectory() as tmpdir:
    bids_root = Path(tmpdir) / "bids_batch"

    make_bids_dataset_description(
        bids_root=bids_root,
        name="eegbci-motor-imagery-demo",
        authors=["BPN Lab"],
    )

    # subject_runs: list of dicts, one entry per run
    subject_runs = [
        {"raw": raws[0], "subject": "01", "session": "01", "run": "06"},
        {"raw": raws[1], "subject": "01", "session": "01", "run": "10"},
    ]

    bids_paths = batch_export_to_bids(
        subject_runs=subject_runs,
        bids_root=bids_root,
        task="motorImagery",
        line_freq=LINE_FREQ,
        overwrite=True,
        verbose=False,
    )

    print(f"\nBatch export: {len(bids_paths)} runs written")
    for bp in bids_paths:
        print(" ", bp.fpath)
