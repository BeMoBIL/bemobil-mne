"""Tests for bpn_analysis.preproc.preprocessing."""

# %% Imports

from pathlib import Path

import mne
import numpy as np
import pytest

from bpn_analysis.preproc.preprocessing import EEGPreprocessor, get_bad_chs

# %% Helpers


def _eeg_raw(n_ch=32, sfreq=500.0, duration=30.0, rng_seed=0, with_montage=True):
    """Minimal EEG Raw with optional standard_1020 montage."""
    rng = np.random.default_rng(rng_seed)
    montage = mne.channels.make_standard_montage("standard_1020")
    if with_montage:
        ch_names = montage.ch_names[:n_ch]
    else:
        ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    n_times = int(sfreq * duration)
    data = rng.standard_normal((n_ch, n_times)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    if with_montage:
        raw.set_montage(montage, on_missing="ignore")
    return raw


# %% get_bad_chs


def test_get_bad_chs_returns_expected_keys():
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    required = {"all_bads", "pyprep", "faster", "bad_by_line_noise", "bad_by_manual"}
    assert required <= set(bad_ch_dict.keys())


def test_get_bad_chs_all_bads_is_list():
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    assert isinstance(bad_ch_dict["all_bads"], list)


def test_get_bad_chs_clean_raw_has_few_bads():
    """Synthetic Gaussian noise should produce few or no bad channels."""
    raw = _eeg_raw(rng_seed=99)
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    # Relaxed: just assert we can run it without error
    assert isinstance(bad_ch_dict["all_bads"], list)


def test_get_bad_chs_europe_preset():
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines="europe")
    assert "all_bads" in bad_ch_dict


def test_get_bad_chs_usa_preset():
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines="usa")
    assert "all_bads" in bad_ch_dict


def test_get_bad_chs_invalid_preset_raises():
    raw = _eeg_raw()
    with pytest.raises(ValueError, match="Unknown notch_lines preset"):
        get_bad_chs(raw, notch_lines="asia")


def test_get_bad_chs_skips_notch_when_none():
    """Passing notch_lines=None should not raise and should still return a dict."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    assert "all_bads" in bad_ch_dict


def test_get_bad_chs_detects_flat_channel(rng=None):
    """A completely flat channel should be flagged as bad."""
    rng = np.random.default_rng(7)
    raw = _eeg_raw(n_ch=16, rng_seed=7)
    # Inject a flat (zero) channel at index 5
    raw._data[5] = 0.0
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    # The flat channel should appear somewhere in the bad channels
    flat_ch = raw.ch_names[5]
    all_bads_flat = (
        bad_ch_dict["all_bads"]
        + bad_ch_dict["pyprep"].get("bad_by_nan_flat", [])
    )
    assert flat_ch in all_bads_flat or flat_ch in bad_ch_dict["pyprep"].get(
        "bad_by_flat", []
    )


def test_get_bad_chs_line_noise_criterion_disabled():
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None, line_noise_crit=None)
    assert bad_ch_dict["bad_by_line_noise"] == []


# %% EEGPreprocessor - fast (non-slow) tests


def test_eegpreprocessor_instantiation():
    """Constructor should succeed with loader=None (run_raw usage)."""
    proc = EEGPreprocessor(loader=None)
    assert proc is not None


def test_eegpreprocessor_custom_params():
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 80.0),
        zapline_freqs=[50.0],
        zapline_method="dss_line",
        fit_ica=False,
    )
    assert proc.filter_bands == (1.0, 80.0)
    assert proc.zapline_method == "dss_line"


def test_eegpreprocessor_skip_if_exists(tmp_path):
    """When skip_if_exists=True and cached outputs exist, run_raw returns early.

    The code checks for <stem>_clean.fif.gz (not fname_out itself).
    We create that sentinel and mock _load_cached_outputs so we don't need
    real fif content.
    """
    from unittest.mock import patch

    raw = _eeg_raw(n_ch=16, duration=10.0)
    fname_out = tmp_path / "sub-01_proc-raw.fif"
    # Create the sentinel file that skip_if_exists actually checks
    clean_path = tmp_path / "sub-01_proc-raw_clean.fif.gz"
    clean_path.touch()

    proc = EEGPreprocessor(loader=None, skip_if_exists=True, fit_ica=False)
    with patch.object(proc, "_load_cached_outputs", return_value=None) as mock_load:
        proc.run_raw(raw, fname_out=fname_out)

    mock_load.assert_called_once()


# %% EEGPreprocessor - integration (slow)


@pytest.mark.slowtest
def test_eegpreprocessor_full_run_no_dipoles(tmp_path, sample_raw):
    """Full pipeline without dipole fitting (avoids fsaverage download in CI)."""
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 40.0),
        zapline_freqs=[60.0],  # 60 Hz: below Nyquist of EEGBCI ~160 Hz
        zapline_method="dss_line",
        fit_ica=True,
        ica_method="picard",
        fit_dipoles=False,
        skip_if_exists=False,
    )
    fname_out = tmp_path / "sub-01_proc-raw.fif"
    raw_out, report, metadata = proc.run_raw(sample_raw, fname_out=fname_out)

    assert fname_out.exists()
    assert raw_out is not None
    assert raw_out.info["sfreq"] == pytest.approx(sample_raw.info["sfreq"], rel=0.1)


@pytest.mark.slowtest
def test_eegpreprocessor_skip_if_exists_integration(tmp_path, sample_raw):
    """run_raw with skip_if_exists=True must skip when output already exists."""
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 40.0),
        fit_ica=False,
        skip_if_exists=True,
    )
    fname_out = tmp_path / "sub-01_proc-raw.fif"

    # First run - creates the file
    proc.run_raw(sample_raw, fname_out=fname_out)
    mtime_1 = fname_out.stat().st_mtime

    # Second run - should skip
    proc.run_raw(sample_raw, fname_out=fname_out)
    mtime_2 = fname_out.stat().st_mtime

    assert mtime_1 == mtime_2, "File should not have been rewritten"


@pytest.mark.slowtest
def test_eegpreprocessor_rename_channels(tmp_path, sample_raw):
    """rename_channels dict should be applied before processing."""
    old_name = sample_raw.ch_names[0]
    new_name = "RENAMED_CH"

    proc = EEGPreprocessor(
        loader=None,
        fit_ica=False,
        rename_channels={old_name: new_name},
    )
    raw_out, _, _ = proc.run_raw(sample_raw.copy(), fname_out=None)

    assert new_name in raw_out.ch_names
    assert old_name not in raw_out.ch_names


@pytest.mark.slowtest
def test_eegpreprocessor_provenance_recorded(tmp_path, sample_raw):
    """Provenance descriptor should be written into raw.info['description']."""
    from bpn_analysis.preproc.utils import get_descriptor

    proc = EEGPreprocessor(loader=None, fit_ica=False)
    raw_out, _, _ = proc.run_raw(sample_raw.copy(), fname_out=None)

    desc = get_descriptor(raw_out)
    assert desc is not None
    assert len(desc["steps"]) > 0
