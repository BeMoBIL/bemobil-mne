"""Tests for bemobil_mne.preproc.epoching."""

# %% Imports

import logging

import mne
import numpy as np

from bemobil_mne.preproc import EpochPreparer, get_stimulus_rename_map

# %% Helpers


def _raw_with_trials(rng, duration=40.0, sfreq=100.0):
    """EEG with alternating A/x and B/y trialStart annotations every 2 s."""
    n_times = int(duration * sfreq)
    info = mne.create_info(["Cz", "Pz"], sfreq, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((2, n_times)) * 1e-6, info)
    onsets = np.arange(2, duration - 2, 2.0)
    desc = [
        "trialStart;condition:A;stimulus:x"
        if i % 2
        else "trialStart;condition:B;stimulus:y"
        for i in range(len(onsets))
    ]
    raw.set_annotations(mne.Annotations(onsets, 0, desc))
    return raw


def _preparer(**kwargs):
    remaps = {"c": {"A/x": "target", "B/y": "standard"}}
    defaults = dict(epoch_tlimes=(-0.2, 0.5), baseline=(-0.2, 0), tshift=0)
    return EpochPreparer(remaps, **{**defaults, **kwargs})


# %% Tests


def test_get_stimulus_rename_map():
    """Parse condition/stimulus labels and assign sorted integer IDs."""
    rename, ids = get_stimulus_rename_map(
        [
            "trialStart;condition:A;stimulus:x",
            "other",
            "trialStart;condition:B;stimulus:y",
        ]
    )
    assert rename == {
        "trialStart;condition:A;stimulus:x": "A/x",
        "trialStart;condition:B;stimulus:y": "B/y",
    }
    assert ids == {"A/x": 1, "B/y": 2}


def test_epoch_preparer_rejects_epochs_in_bad_segments(rng, tmp_path):
    """Epochs overlapping BAD_ annotations are dropped."""
    raw = _raw_with_trials(rng)
    n_target = sum("condition:A" in d for d in raw.annotations.description)
    raw.annotations.append(11.0, 6.0, "BAD_segment")  # covers targets at 12, 16 s
    out = _preparer().run(raw, tmp_path / "sub-01", "c")
    assert len(out["target"]) == n_target - 2
    assert (tmp_path / "sub-01_epo.fif.gz").exists()


def test_epoch_preparer_tshift_only_moves_events(rng, tmp_path):
    """The tshift parameter moves event onsets but not BAD_ segments."""
    raw = _raw_with_trials(rng)
    n_target = sum("condition:A" in d for d in raw.annotations.description)
    # Ends just before the shifted epoch window of the 12 s target
    # ([11.86, 12.56] s); shifting this segment as well would reject it.
    raw.annotations.append(11.80, 0.05, "BAD_segment")
    out = _preparer(tshift=0.06).run(raw, tmp_path / "x", "c")
    epochs = out["target"]
    assert len(epochs) == n_target
    onsets = epochs.events[:, 0] / epochs.info["sfreq"]
    np.testing.assert_allclose(onsets[:2], [4.06, 8.06], atol=0.011)


def test_epoch_preparer_skips_missing_labels(rng, tmp_path, caplog):
    """Labels without events are logged and skipped, not a KeyError."""
    raw = _raw_with_trials(rng)
    prep = EpochPreparer(
        {"c": {"A/x": "target", "C/z": "never"}},
        epoch_tlimes=(-0.2, 0.5),
        baseline=(-0.2, 0),
        tshift=0,
    )
    with caplog.at_level(logging.WARNING):
        out = prep.run(raw, tmp_path / "x.fif", "c")
    assert list(out) == ["target"]
    assert "never" in caplog.text
    assert (tmp_path / "x_epo.fif.gz").exists()
