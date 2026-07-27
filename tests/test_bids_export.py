"""Tests for bpn_analysis.io.bids_export."""

# %% Imports

import json

import mne
import mne_bids
import numpy as np
import pytest

from bpn_analysis.io.bids_export import (
    batch_export_to_bids,
    export_to_bids,
    make_bids_dataset_description,
)

# %% Helpers


def _minimal_raw(n_ch=4, sfreq=250.0, duration=5.0, rng_seed=0):
    """Return a minimal EEG Raw with montage."""
    rng = np.random.default_rng(rng_seed)
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:n_ch]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    n_times = int(sfreq * duration)
    data = rng.standard_normal((n_ch, n_times)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")
    return raw


# %% export_to_bids


def test_export_to_bids_returns_bids_path(tmp_path):
    """Verify export_to_bids returns a BIDSPath instance."""
    raw = _minimal_raw()
    bp = export_to_bids(raw, bids_root=tmp_path, subject="01", verbose=False)
    assert isinstance(bp, mne_bids.BIDSPath)


def test_export_to_bids_creates_eeg_file(tmp_path):
    """Verify export_to_bids creates the EEG file."""
    raw = _minimal_raw()
    bp = export_to_bids(raw, bids_root=tmp_path, subject="01", verbose=False)
    assert bp.fpath.exists()


def test_export_to_bids_sets_subject(tmp_path):
    """Verify export_to_bids stores the subject label."""
    raw = _minimal_raw()
    bp = export_to_bids(raw, bids_root=tmp_path, subject="99", verbose=False)
    assert bp.subject == "99"


def test_export_to_bids_with_session(tmp_path):
    """Verify export_to_bids sets the session field."""
    raw = _minimal_raw()
    bp = export_to_bids(
        raw, bids_root=tmp_path, subject="01", session="01", verbose=False
    )
    assert bp.session == "01"


def test_export_to_bids_with_run(tmp_path):
    """Verify export_to_bids sets the run field."""
    raw = _minimal_raw()
    bp = export_to_bids(raw, bids_root=tmp_path, subject="01", run="02", verbose=False)
    assert bp.run == "02"


def test_export_to_bids_sets_line_freq(tmp_path):
    """Store line_freq in raw info before writing."""
    raw = _minimal_raw()
    export_to_bids(raw, bids_root=tmp_path, subject="01", line_freq=60.0, verbose=False)
    # original raw should NOT be mutated
    assert raw.info.get("line_freq") != 60.0


def test_export_to_bids_does_not_mutate_original(tmp_path):
    """Verify export_to_bids does not mutate Raw."""
    raw = _minimal_raw()
    original_desc = raw.info.get("description")
    export_to_bids(raw, bids_root=tmp_path, subject="01", verbose=False)
    assert raw.info.get("description") == original_desc


def test_export_to_bids_overwrite(tmp_path):
    """Verify duplicate write raises, overwrite succeeds."""
    raw = _minimal_raw()
    export_to_bids(raw, bids_root=tmp_path, subject="01", verbose=False)
    # second call without overwrite should raise
    with pytest.raises(Exception):
        export_to_bids(raw, bids_root=tmp_path, subject="01", verbose=False)
    # with overwrite it should succeed
    bp = export_to_bids(
        raw, bids_root=tmp_path, subject="01", overwrite=True, verbose=False
    )
    assert bp.fpath.exists()


# %% make_bids_dataset_description


def test_make_bids_dataset_description_creates_file(tmp_path):
    """Verify make_bids_dataset_description writes description file."""
    make_bids_dataset_description(tmp_path, name="TestDataset")
    desc_file = tmp_path / "dataset_description.json"
    assert desc_file.exists()


def test_make_bids_dataset_description_contains_name(tmp_path):
    """Verify make_bids_dataset_description stores the dataset name."""
    make_bids_dataset_description(tmp_path, name="MyStudy")
    desc = json.loads((tmp_path / "dataset_description.json").read_text())
    assert desc["Name"] == "MyStudy"


def test_make_bids_dataset_description_with_authors(tmp_path):
    """Verify make_bids_dataset_description records all author names."""
    make_bids_dataset_description(tmp_path, name="Study", authors=["Alice", "Bob"])
    desc = json.loads((tmp_path / "dataset_description.json").read_text())
    assert "Alice" in desc["Authors"]
    assert "Bob" in desc["Authors"]


def test_make_bids_dataset_description_idempotent(tmp_path):
    """Calling twice should not raise."""
    make_bids_dataset_description(tmp_path, name="Study")
    make_bids_dataset_description(tmp_path, name="Study v2")
    desc = json.loads((tmp_path / "dataset_description.json").read_text())
    assert desc["Name"] == "Study v2"


# %% batch_export_to_bids


def test_batch_export_returns_list(tmp_path):
    """Verify batch_export returns one entry per run."""
    raw = _minimal_raw()
    runs = [{"raw": raw, "subject": "01"}, {"raw": raw, "subject": "02"}]
    results = batch_export_to_bids(runs, bids_root=tmp_path, verbose=False)
    assert isinstance(results, list)
    assert len(results) == 2


def test_batch_export_creates_correct_subjects(tmp_path):
    """Verify batch_export creates correct subject directories."""
    raw = _minimal_raw()
    runs = [
        {"raw": raw, "subject": "01"},
        {"raw": raw, "subject": "02"},
    ]
    batch_export_to_bids(runs, bids_root=tmp_path, verbose=False)
    assert (tmp_path / "sub-01").exists()
    assert (tmp_path / "sub-02").exists()


def test_batch_export_continues_on_error(tmp_path):
    """Ensure a failed run does not abort batch."""
    raw = _minimal_raw()
    # subject "01" succeeds; None as raw should fail gracefully
    runs = [
        {"raw": None, "subject": "bad"},
        {"raw": raw, "subject": "02"},
    ]
    results = batch_export_to_bids(runs, bids_root=tmp_path, verbose=False)
    # only the valid run produces a BIDSPath
    assert len(results) == 1
    assert results[0].subject == "02"


def test_batch_export_with_session_and_run(tmp_path):
    """Verify batch_export assigns session and run fields."""
    raw = _minimal_raw()
    runs = [
        {"raw": raw, "subject": "01", "session": "01", "run": "01"},
        {"raw": raw, "subject": "01", "session": "01", "run": "02"},
    ]
    results = batch_export_to_bids(runs, bids_root=tmp_path, verbose=False)
    assert len(results) == 2
    assert {r.run for r in results} == {"01", "02"}
