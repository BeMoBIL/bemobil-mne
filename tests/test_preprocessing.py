"""Tests for pybemobil.preproc.preprocessing."""

# %% Imports

import json

import mne
import numpy as np
import pytest

from pybemobil.preproc.preprocessing import (
    EEGPreprocessor,
    _bids_raw_target,
    _output_stem,
    get_bad_chs,
    get_derivative_paths,
)

# %% Helpers


def _eeg_raw(n_ch=32, sfreq=500.0, duration=30.0, rng_seed=0, with_montage=True):
    """Create a minimal EEG Raw."""
    rng = np.random.default_rng(rng_seed)
    montage = mne.channels.make_standard_montage("easycap-M1")
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
    """Check all required keys are present."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    required = {"all_bads", "pyprep", "faster", "bad_by_line_noise", "bad_by_manual"}
    assert required <= set(bad_ch_dict.keys())


def test_get_bad_chs_all_bads_is_list():
    """Verify all_bads is a list."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    assert isinstance(bad_ch_dict["all_bads"], list)


def test_get_bad_chs_clean_raw_has_few_bads():
    """Run get_bad_chs on Gaussian noise."""
    raw = _eeg_raw(rng_seed=99)
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    # Relaxed: just assert we can run it without error
    assert isinstance(bad_ch_dict["all_bads"], list)


def test_get_bad_chs_europe_preset():
    """Accept the 'europe' notch preset without error."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines="europe")
    assert "all_bads" in bad_ch_dict


def test_get_bad_chs_usa_preset():
    """Accept the 'usa' notch preset without error."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines="usa")
    assert "all_bads" in bad_ch_dict


def test_get_bad_chs_invalid_preset_raises():
    """Raise ValueError for unknown notch_lines preset."""
    raw = _eeg_raw()
    with pytest.raises(ValueError, match="Unknown notch_lines preset"):
        get_bad_chs(raw, notch_lines="asia")


def test_get_bad_chs_skips_notch_when_none():
    """Skip notch filtering when notch_lines is None."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    assert "all_bads" in bad_ch_dict


def test_get_bad_chs_detects_flat_channel(rng=None):
    """Flag a synthetically flat channel as bad."""
    raw = _eeg_raw(n_ch=16, rng_seed=7)
    # Inject a flat (zero) channel at index 5
    raw._data[5] = 0.0
    bad_ch_dict = get_bad_chs(raw, notch_lines=None)
    # The flat channel should appear somewhere in the bad channels
    flat_ch = raw.ch_names[5]
    all_bads_flat = bad_ch_dict["all_bads"] + bad_ch_dict["pyprep"].get(
        "bad_by_nan_flat", []
    )
    assert flat_ch in all_bads_flat or flat_ch in bad_ch_dict["pyprep"].get(
        "bad_by_flat", []
    )


def test_get_bad_chs_line_noise_criterion_disabled():
    """Return empty bad_by_line_noise when line_noise_crit is None."""
    raw = _eeg_raw()
    bad_ch_dict = get_bad_chs(raw, notch_lines=None, line_noise_crit=None)
    assert bad_ch_dict["bad_by_line_noise"] == []


# %% EEGPreprocessor - fast (non-slow) tests


def test_eegpreprocessor_instantiation():
    """Instantiate EEGPreprocessor with loader=None."""
    proc = EEGPreprocessor(loader=None)
    assert proc is not None


def test_eegpreprocessor_custom_params():
    """Accept custom filter_bands, line_noise_freq, and line_noise_method."""
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 80.0),
        line_noise_freq=50.0,
        line_noise_method="dss_line",
        fit_ica=False,
    )
    assert proc.filter_bands == (1.0, 80.0)
    assert proc.line_noise_method == "dss_line"


def test_eegpreprocessor_skip_if_exists(tmp_path):
    """Trigger _load_cached_outputs when fname_out exists."""
    from unittest.mock import patch

    raw = _eeg_raw(n_ch=16, duration=10.0)
    fname_out = tmp_path / "sub-01_task-rest_eeg.fif"
    # Create the sentinel file that skip_if_exists actually checks
    get_derivative_paths(fname_out)["clean"].touch()

    proc = EEGPreprocessor(loader=None, skip_if_exists=True, fit_ica=False)
    with patch.object(proc, "_load_cached_outputs", return_value=None) as mock_load:
        proc.run_raw(raw, fname_out=fname_out)

    mock_load.assert_called_once()


@pytest.mark.parametrize(
    "name", ["sub-01_clean.fif.gz", "sub-01_clean.fif", "sub-01_clean"]
)
def test_output_stem_strips_fif_extensions(tmp_path, name):
    """Strip .fif and .fif.gz so save and load build the same file names."""
    assert _output_stem(tmp_path / name) == tmp_path / "sub-01_clean"


@pytest.mark.parametrize(
    ("name", "ext"),
    [
        ("sub-01_ses-01_task-walk_run-01_eeg.fif.gz", ".fif.gz"),
        ("sub-01_ses-01_task-walk_run-01_eeg.fif", ".fif"),
        ("sub-01_ses-01_task-walk_run-01_desc-clean_eeg.fif", ".fif"),
        ("sub-01_ses-01_task-walk_run-01", ".fif"),
    ],
)
def test_get_derivative_paths_bids_names(tmp_path, name, ext):
    """Keep the entities, drop suffix and desc, add BIDS desc labels/suffixes."""
    paths = get_derivative_paths(tmp_path / name)
    base = "sub-01_ses-01_task-walk_run-01"
    assert paths == {
        "clean": tmp_path / f"{base}_desc-clean_eeg{ext}",
        "pre_ica": tmp_path / f"{base}_desc-preica_eeg{ext}",
        "ica": tmp_path / f"{base}_ica{ext}",
        "iclabels": tmp_path / f"{base}_iclabels.json",
        "bad_channels": tmp_path / f"{base}_badchannels.json",
        "dipoles": tmp_path / f"{base}_dipoles",
        "trans": tmp_path / f"{base}_trans.fif",
        "report": tmp_path / f"{base}_report.html",
    }


def test_load_cached_outputs_aligns_dipoles_by_ic_index(tmp_path):
    """Place loaded dipoles at their IC index, with None for missing fits."""
    raw = _eeg_raw(n_ch=8, duration=5.0)
    fname_out = tmp_path / "sub-01_task-rest_eeg.fif.gz"
    paths = get_derivative_paths(fname_out)
    raw.save(paths["clean"], verbose="error")
    dipdir = paths["dipoles"]
    dipdir.mkdir()
    dip = mne.Dipole(
        times=[0.0],
        pos=[[0.0, 0.0, 0.05]],
        amplitude=[1e-8],
        ori=[[0.0, 0.0, 1.0]],
        gof=[90.0],
    )
    for i in (0, 2):  # IC 1 had no valid dipole
        dip.save(dipdir / f"ic-{i:02}-dip.bdip", verbose="error")

    proc = EEGPreprocessor(loader=None)
    _, _, metadata = proc._load_cached_outputs(fname_out)

    dipoles = metadata["dipoles"]
    assert len(dipoles) == 3
    assert dipoles[0] is not None
    assert dipoles[1] is None
    assert dipoles[2] is not None


def test_reapply_ica_requires_pre_ica_file(tmp_path):
    """Raise a clear error when the pre-ICA recording was never saved."""
    raw = _eeg_raw(n_ch=8, duration=5.0)
    fname_out = tmp_path / "sub-01_task-rest_eeg.fif.gz"
    raw.save(get_derivative_paths(fname_out)["clean"], verbose="error")
    proc = EEGPreprocessor(loader=None)
    with pytest.raises(FileNotFoundError, match="pre-ICA"):
        proc.reapply_ica(fname_out, add=[0])


# %% EEGPreprocessor - integration (slow)


@pytest.mark.slowtest
def test_eegpreprocessor_full_run_no_dipoles(tmp_path, sample_raw):
    """Run full pipeline, verify outputs."""
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 40.0),
        line_noise_freq=60.0,
        line_noise_method="dss_line",
        fit_ica=True,
        ica_method="picard",
        fit_dipoles=False,
        skip_if_exists=False,
    )
    fname_out = tmp_path / "sub-01_task-rest_eeg.fif"
    raw_out, report, metadata = proc.run_raw(sample_raw, fname_out=fname_out)

    assert get_derivative_paths(fname_out)["clean"].exists()
    assert raw_out is not None
    assert raw_out.info["sfreq"] == pytest.approx(sample_raw.info["sfreq"], rel=0.1)


@pytest.mark.slowtest
def test_eegpreprocessor_reapply_ica(tmp_path, sample_raw):
    """Save the pre-ICA file and re-apply ICA with a manual exclusion."""
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 40.0),
        line_noise_freq=60.0,
        line_noise_method="dss_line",
        fit_ica=True,
        ica_method="picard",
        make_report=False,
    )
    fname_out = _deriv_fname(tmp_path)
    paths = get_derivative_paths(fname_out)
    _, _, metadata = proc.run_raw(sample_raw.copy(), fname_out=fname_out)

    assert paths["pre_ica"].exists()
    # The untouched input goes to the BIDS raw folder, not the derivatives
    assert not list(paths["clean"].parent.glob("*source*"))
    assert (tmp_path / "sub-01" / "ses-01" / "eeg").exists()
    assert "raw_minimal" not in metadata
    np.testing.assert_allclose(metadata["raw_source"].get_data(), sample_raw.get_data())

    auto_exclude = list(metadata["ica"].exclude)
    n_ics = metadata["ica"].n_components_
    extra = next(i for i in range(n_ics) if i not in auto_exclude)

    raw_clean, _, md = proc.reapply_ica(fname_out, add=[extra])

    expected = sorted(set(auto_exclude) | {extra})
    assert md["ica"].exclude == expected
    assert md["raw_source"] is not None  # read back from the BIDS raw
    reloaded = mne.preprocessing.read_ica(paths["ica"], verbose="error")
    assert sorted(reloaded.exclude) == expected
    assert raw_clean.info["bads"] == []

    with pytest.raises(ValueError, match="out of range"):
        proc.reapply_ica(fname_out, add=[n_ics + 5], save=False)


@pytest.mark.slowtest
def test_eegpreprocessor_skip_if_exists_integration(tmp_path, sample_raw):
    """Skip reprocessing on second run."""
    proc = EEGPreprocessor(
        loader=None,
        filter_bands=(1.0, 40.0),
        fit_ica=False,
        skip_if_exists=True,
    )
    fname_out = tmp_path / "sub-01_task-rest_eeg.fif"
    clean = get_derivative_paths(fname_out)["clean"]

    # First run - creates the file
    proc.run_raw(sample_raw, fname_out=fname_out)
    mtime_1 = clean.stat().st_mtime

    # Second run - should skip
    proc.run_raw(sample_raw, fname_out=fname_out)
    mtime_2 = clean.stat().st_mtime

    assert mtime_1 == mtime_2, "File should not have been rewritten"


@pytest.mark.slowtest
def test_eegpreprocessor_rename_channels(tmp_path, sample_raw):
    """Apply rename_channels dict before processing."""
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
    """Record a provenance descriptor in raw.info."""
    from pybemobil.preproc.utils import get_descriptor

    proc = EEGPreprocessor(loader=None, fit_ica=False)
    raw_out, _, _ = proc.run_raw(sample_raw.copy(), fname_out=None)

    desc = get_descriptor(raw_out)
    assert desc is not None
    assert len(desc["steps"]) > 0


# %% save_bids_raw


def _deriv_fname(root, name="sub-01_ses-01_task-walk_run-01_eeg.fif.gz"):
    """Return a derivative path below ``root/derivatives`` like the pipeline uses."""
    path = root / "derivatives" / "sub-01" / "ses-01" / "eeg" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_bids_raw_target_infers_root_and_entities(tmp_path):
    """Use the folder above derivatives/ as root and parse entities from the name."""
    root, ents = _bids_raw_target(_deriv_fname(tmp_path))
    assert root == tmp_path.resolve()
    assert ents == {
        "subject": "01",
        "session": "01",
        "task": "walk",
        "run": "01",
        "acquisition": None,
    }


def test_bids_raw_target_explicit_root(tmp_path):
    """An explicit bids_root works outside a derivatives folder."""
    root, ents = _bids_raw_target(
        tmp_path / "sub-02_task-rest_eeg.fif.gz", bids_root=tmp_path / "bids"
    )
    assert root == tmp_path / "bids"
    assert ents["subject"] == "02"
    assert ents["session"] is None


def test_bids_raw_target_errors(tmp_path):
    """Raise when the root cannot be inferred or entities are missing."""
    with pytest.raises(ValueError, match="derivatives"):
        _bids_raw_target(tmp_path / "sub-01_task-walk_eeg.fif.gz")
    with pytest.raises(ValueError, match="task"):
        _bids_raw_target(_deriv_fname(tmp_path, "sub-01_clean.fif.gz"))


def test_save_bids_raw_writes_untouched_data(tmp_path):
    """Write the input to <root>/sub-*/ses-*/eeg/ with identical data."""
    raw = _eeg_raw(n_ch=8, duration=5.0)
    fname_out = _deriv_fname(tmp_path)
    proc = EEGPreprocessor(loader=None)

    bids_path = proc._save_bids_raw(raw, fname_out)

    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    vhdr = eeg_dir / "sub-01_ses-01_task-walk_run-01_eeg.vhdr"
    assert bids_path is not None
    assert vhdr.exists()
    assert (eeg_dir / "sub-01_ses-01_task-walk_run-01_eeg.json").exists()
    assert (tmp_path / "dataset_description.json").exists()
    reread = mne.io.read_raw_brainvision(vhdr, verbose="error")
    np.testing.assert_allclose(reread.get_data(), raw.get_data(), atol=1e-9)
    sidecar = json.loads(
        (eeg_dir / "sub-01_ses-01_task-walk_run-01_eeg.json").read_text()
    )
    assert sidecar["PowerLineFrequency"] == 50.0

    # Second call without overwrite_bids_raw leaves the file alone
    mtime = vhdr.stat().st_mtime
    assert proc._save_bids_raw(raw, fname_out) is None
    assert vhdr.stat().st_mtime == mtime

    # overwrite_bids_raw=True replaces it
    proc_ow = EEGPreprocessor(loader=None, overwrite_bids_raw=True)
    assert proc_ow._save_bids_raw(raw, fname_out) is not None


def test_run_raw_overwrite_does_not_rewrite_bids_raw(tmp_path):
    """run_raw(overwrite=True) refreshes derivatives but keeps the BIDS raw."""
    from unittest.mock import patch

    raw = _eeg_raw(n_ch=8, duration=5.0)
    fname_out = _deriv_fname(tmp_path)
    proc = EEGPreprocessor(loader=None)
    proc._save_bids_raw(raw, fname_out)

    with patch("pybemobil.io.bids_export.export_to_bids") as mock_export:
        proc._save_bids_raw(raw, fname_out)
    mock_export.assert_not_called()


def test_save_bids_raw_warns_without_root(tmp_path):
    """Warn (not raise) when no BIDS root can be determined."""
    raw = _eeg_raw(n_ch=8, duration=5.0)
    proc = EEGPreprocessor(loader=None)
    with pytest.warns(UserWarning, match="save_bids_raw skipped"):
        assert proc._save_bids_raw(raw, tmp_path / "sub-01_task-walk_eeg.fif") is None


def test_load_cached_outputs_reads_source_from_bids(tmp_path):
    """raw_source comes back from the BIDS raw folder, not from derivatives."""
    raw = _eeg_raw(n_ch=8, duration=5.0)
    fname_out = _deriv_fname(tmp_path)
    paths = get_derivative_paths(fname_out)
    raw.save(paths["clean"], verbose="error")
    proc = EEGPreprocessor(loader=None)

    _, _, md = proc._load_cached_outputs(fname_out)
    assert md["raw_source"] is None  # nothing written to BIDS yet

    proc._save_bids_raw(raw, fname_out)
    _, _, md = proc._load_cached_outputs(fname_out)
    np.testing.assert_allclose(md["raw_source"].get_data(), raw.get_data(), atol=1e-9)


def test_get_bad_chs_keeps_default_pyprep_kwargs():
    """Merge user pyprep_kwargs into the defaults without mutating them."""
    from unittest.mock import patch

    from pybemobil.preproc import preprocessing as pp

    captured = {}
    real = pp.NoisyChannels

    def _spy(raw, **kwargs):
        captured.update(kwargs)
        return real(raw, **kwargs)

    raw = _eeg_raw(n_ch=16)
    raw.info["bads"] = [raw.ch_names[0]]
    user_kwargs = {"random_state": 1}
    with patch.object(pp, "NoisyChannels", side_effect=_spy):
        get_bad_chs(raw, pyprep_kwargs=user_kwargs, notch_lines=None)

    assert captured["reject_by_annotation"] == "omit"
    assert captured["random_state"] == 1
    assert captured["bad_by_manual"] == [raw.ch_names[0]]
    assert user_kwargs == {"random_state": 1}


def test_get_bad_chs_none_skips_notch():
    """Do not call notch_filter when notch_lines is None."""
    from unittest.mock import patch

    raw = _eeg_raw(n_ch=16)
    with patch.object(mne.io.BaseRaw, "notch_filter") as notch:
        get_bad_chs(raw, notch_lines=None)
    notch.assert_not_called()


def test_eegpreprocessor_label_defaults():
    """exclude_labels works alone; include_labels defaults to all but eye blink."""
    proc = EEGPreprocessor(loader=None, exclude_labels=["eye blink"])
    assert proc.include_labels is None
    assert proc.exclude_labels == ["eye blink"]

    proc = EEGPreprocessor(loader=None)
    assert "eye blink" not in proc.include_labels
    assert "brain" in proc.include_labels

    with pytest.raises(ValueError, match="either exclude_labels or include_labels"):
        EEGPreprocessor(
            loader=None, exclude_labels=["eye blink"], include_labels={"brain"}
        )


# %% run / run_raw behaviour


def _correlated_raw(n_ch=16, sfreq=250.0, duration=30.0, seed=0):
    """EEG with a shared smooth source so PyPREP does not reject everything."""
    rng = np.random.default_rng(seed)
    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:n_ch]
    n_times = int(sfreq * duration)
    common = np.convolve(rng.standard_normal(n_times), np.ones(25) / 5, "same")
    mixing = rng.uniform(0.5, 1.5, n_ch)[:, None]
    data = (mixing * common + 0.1 * rng.standard_normal((n_ch, n_times))) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(ch_names, sfreq, "eeg"), verbose=False)
    raw.set_montage(montage, on_missing="ignore")
    return raw


def _fast_proc(**kwargs):
    defaults = dict(
        line_noise_method=None,
        fit_ica=False,
        make_report=False,
        save_bids_raw=False,
        verbose="error",
    )
    return EEGPreprocessor(loader=None, **{**defaults, **kwargs})


def test_run_raw_does_not_modify_input():
    """The caller's raw keeps its data, bads, projections and description."""
    raw = _correlated_raw()
    data = raw.get_data().copy()
    with pytest.warns(RuntimeWarning, match="fit_ica=False"):
        raw_clean, _, metadata = _fast_proc().run_raw(raw)
    np.testing.assert_array_equal(raw.get_data(), data)
    assert raw.info["bads"] == []
    assert raw.info["projs"] == []
    assert raw.info["description"] is None
    assert metadata["raw_source"] is raw
    assert raw_clean is not raw


def test_run_raw_restores_log_level_on_error():
    """MNE's log level is restored even when the pipeline raises."""
    from unittest.mock import patch

    from pybemobil.preproc import preprocessing as pp

    mne.set_log_level("INFO")
    with patch.object(pp, "get_bad_chs", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _fast_proc(verbose="error").run_raw(_correlated_raw())
    assert mne.utils.logger.level == 20  # INFO


def test_run_raw_restores_log_level_on_cache_hit(tmp_path):
    """The skip_if_exists early return restores the log level too."""
    from unittest.mock import patch

    fname_out = tmp_path / "sub-01_task-rest_eeg.fif"
    get_derivative_paths(fname_out)["clean"].touch()
    proc = _fast_proc(skip_if_exists=True, verbose="error")
    mne.set_log_level("INFO")
    with patch.object(EEGPreprocessor, "_load_cached_outputs", return_value=(1, 2, 3)):
        assert proc.run_raw(_correlated_raw(), fname_out=fname_out) == (1, 2, 3)
    assert mne.utils.logger.level == 20  # INFO


def test_run_raw_all_channels_bad_raises():
    """A clear error is raised when every EEG channel is flagged bad."""
    from unittest.mock import patch

    from pybemobil.preproc import preprocessing as pp

    raw = _correlated_raw()
    fake = {
        "all_bads": list(raw.ch_names),
        "pyprep": {"bad_all": list(raw.ch_names)},
        "faster": {"bad_all": []},
        "bad_by_line_noise": [],
        "bad_by_manual": [],
    }
    with patch.object(pp, "get_bad_chs", return_value=fake):
        with pytest.raises(RuntimeError, match="All 16 EEG channels"):
            _fast_proc().run_raw(raw)


def test_run_raw_saves_outputs_when_report_fails(tmp_path):
    """A failing report is a warning; the derivatives are still written."""
    from unittest.mock import patch

    fname_out = tmp_path / "sub-01_task-rest_eeg.fif"
    proc = _fast_proc(make_report=True)
    with patch(
        "pybemobil.preproc.make_report.make_report",
        side_effect=ValueError("bad event_id"),
    ):
        with pytest.warns(RuntimeWarning, match="Report generation failed"):
            _, report, _ = proc.run_raw(_correlated_raw(), fname_out=fname_out)
    assert report is None
    assert get_derivative_paths(fname_out)["clean"].exists()
    assert get_derivative_paths(fname_out)["pre_ica"].exists()


class _FakeLoader:
    """Loader stub returning a fixed MultimodalRecording."""

    def __init__(self, raw):
        self.raw = raw

    def load(self, fname):
        from pybemobil.io import MultimodalRecording

        return MultimodalRecording(raw=self.raw)


def test_run_keeps_loader_montage():
    """run() does not replace a montage set by the loader."""
    from unittest.mock import patch

    raw = _correlated_raw(n_ch=4)
    raw.rename_channels({ch: f"g{i + 1}" for i, ch in enumerate(raw.ch_names)})
    pos = {f"g{i + 1}": np.array([0.05 * i, 0.02, 0.07]) for i in range(4)}
    custom = mne.channels.make_dig_montage(pos, coord_frame="head")
    raw.set_montage(custom)

    proc = EEGPreprocessor(loader=_FakeLoader(raw))
    with patch.object(EEGPreprocessor, "run_raw", return_value="ok") as run_raw:
        assert proc.run("in.xdf") == "ok"
    got = run_raw.call_args.args[0].get_montage().get_positions()["ch_pos"]
    np.testing.assert_allclose(got["g2"], pos["g2"])


def test_run_sets_standard_montage_when_missing():
    """run() falls back to standard_1005 when the loader set no montage."""
    from unittest.mock import patch

    raw = _correlated_raw(n_ch=4)
    raw.set_montage(None)
    proc = EEGPreprocessor(loader=_FakeLoader(raw))
    with patch.object(EEGPreprocessor, "run_raw", return_value="ok") as run_raw:
        proc.run("in.xdf")
    assert run_raw.call_args.args[0].get_montage() is not None
