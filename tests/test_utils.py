"""Tests for bemobil_mne.preproc.utils."""

# %% Imports

import logging

import mne
import numpy as np
import pytest

from bemobil_mne.preproc.utils import (
    StepTimer,
    append_desc,
    build_sys_info,
    compute_asr,
    compute_mi_reduction,
    compute_zapline,
    detect_bad_by_line_noise,
    format_duration,
    get_descriptor,
    get_raw_subset,
    init_descriptor,
    set_descriptor,
    sig_params,
)

# %% format_duration


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "0.0s"),
        (1.5, "1.5s"),
        (59.9, "59.9s"),
        (60.0, "1m 00.0s"),
        (90.0, "1m 30.0s"),
        (3600.0, "1h 00m 00.0s"),
        (3725.0, "1h 02m 05.0s"),
        (7384.5, "2h 03m 04.5s"),
    ],
)
def test_format_duration(seconds, expected):
    """Convert seconds to human-readable string."""
    assert format_duration(seconds) == expected


# %% StepTimer


def test_step_timer_records_and_totals():
    """Record steps and compute total duration."""
    timer = StepTimer()
    timer.log_step("filter", 2.0)
    timer.log_step("ica", 8.0)

    assert len(timer.timings) == 2
    assert timer.timings[0] == {"name": "filter", "duration_s": 2.0}
    assert timer.total_s == pytest.approx(10.0)


def test_step_timer_logs_each_step(caplog):
    """Log each step name and duration."""
    with caplog.at_level(logging.INFO):
        timer = StepTimer()
        timer.log_step("zapline", 12.0)

    assert "zapline" in caplog.text
    assert "12.0s" in caplog.text


def test_step_timer_format_summary():
    """Include step names, TOTAL, and percentages in summary."""
    timer = StepTimer()
    timer.log_step("filter", 30.0)
    timer.log_step("ica", 90.0)
    summary = timer.format_summary()

    assert "filter" in summary
    assert "ica" in summary
    assert "TOTAL" in summary
    # filter is 25 % of 120 s
    assert "25.0%" in summary


def test_step_timer_empty_total():
    """Return zero total when no steps logged."""
    timer = StepTimer()
    assert timer.total_s == 0.0


# %% Provenance / descriptor helpers


def test_init_descriptor_structure():
    """Return dict with required top-level keys."""
    desc = init_descriptor(source="sub-01.xdf", pipeline="bpn")
    assert set(desc.keys()) >= {"pipeline", "input", "timestamp", "versions", "steps"}
    assert desc["pipeline"] == "bpn"
    assert desc["input"] == "sub-01.xdf"
    assert desc["steps"] == []


def test_init_descriptor_list_source():
    """Serialise list of Path sources as strings."""
    from pathlib import Path

    desc = init_descriptor(source=[Path("a.xdf"), Path("b.xdf")])
    assert desc["input"] == ["a.xdf", "b.xdf"]


def test_init_descriptor_none_source():
    """Store None when no source provided."""
    desc = init_descriptor()
    assert desc["input"] is None


def test_set_get_descriptor_roundtrip(tiny_raw):
    """Round-trip a descriptor through Raw.info."""
    desc = init_descriptor(source="test.xdf", pipeline="test")
    set_descriptor(tiny_raw, desc)
    recovered = get_descriptor(tiny_raw)
    assert recovered is not None
    assert recovered["pipeline"] == "test"
    assert recovered["input"] == "test.xdf"


def test_get_descriptor_missing_returns_none(tiny_raw):
    """Return None when no description stored."""
    # Raw has no description set by default
    result = get_descriptor(tiny_raw)
    assert result is None


def test_get_descriptor_invalid_json(tiny_raw):
    """Return None when description is invalid JSON."""
    tiny_raw.info["description"] = "not-json{{{{"
    assert get_descriptor(tiny_raw) is None


def test_append_desc_creates_descriptor_if_missing(tiny_raw):
    """Initialise descriptor and record step when missing."""
    append_desc(tiny_raw, "filter", l_freq=1.0, h_freq=100.0)
    desc = get_descriptor(tiny_raw)
    assert desc is not None
    assert len(desc["steps"]) == 1
    assert desc["steps"][0]["name"] == "filter"
    assert desc["steps"][0]["l_freq"] == 1.0


def test_append_desc_accumulates_steps(tiny_raw):
    """Append successive steps to existing descriptor."""
    append_desc(tiny_raw, "filter")
    append_desc(tiny_raw, "ica", method="picard")
    desc = get_descriptor(tiny_raw)
    assert len(desc["steps"]) == 2
    assert desc["steps"][1]["name"] == "ica"


# %% sig_params


def test_sig_params_filters_correctly():
    """Return only kwargs accepted by the signature."""

    def fn(a, b, c=3):
        """Accept positional and keyword arguments."""
        pass

    result = sig_params(fn, a=1, b=2, d=99)
    assert result == {"a": 1, "b": 2}
    assert "d" not in result


def test_sig_params_passes_all_when_var_kwargs():
    """Pass all kwargs through var-kwargs functions."""

    def fn(**kwargs):
        """Accept any keyword arguments."""
        pass

    result = sig_params(fn, x=1, y=2)
    assert result == {"x": 1, "y": 2}


# %% get_raw_subset


def test_get_raw_subset_returns_requested(tiny_raw):
    """Return Raw with exactly the requested channels."""
    present = tiny_raw.ch_names[:3]
    sub = get_raw_subset(tiny_raw, subset_chs=present)
    assert sub is not None
    assert sub.ch_names == present


def test_get_raw_subset_skips_missing(tiny_raw):
    """Silently drop channel names not in Raw."""
    chs = [tiny_raw.ch_names[0], "NONEXISTENT_CH"]
    sub = get_raw_subset(tiny_raw, subset_chs=chs)
    assert sub is not None
    assert "NONEXISTENT_CH" not in sub.ch_names


def test_get_raw_subset_returns_none_when_none_present(tiny_raw):
    """Return None when no requested channels exist."""
    sub = get_raw_subset(tiny_raw, subset_chs=["GHOST1", "GHOST2"])
    assert sub is None


# %% compute_zapline


@pytest.fixture
def eeg_with_line_noise(rng):
    """Create 8-ch EEG with 50 Hz line noise."""
    sfreq = 500.0
    n_times = int(sfreq * 20)
    t = np.arange(n_times) / sfreq
    ch_names = [f"EEG{i:03d}" for i in range(8)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    noise = np.sin(2 * np.pi * 50 * t) * 5e-6
    data = rng.standard_normal((8, n_times)) * 1e-6 + noise
    return mne.io.RawArray(data, info, verbose=False)


def test_compute_zapline_dss_line_runs(eeg_with_line_noise):
    """Run dss_line zapline without error, preserve shape."""
    raw_clean = compute_zapline(
        eeg_with_line_noise, noise_freqs=50.0, method="dss_line"
    )
    assert raw_clean.get_data().shape == eeg_with_line_noise.get_data().shape


def test_compute_zapline_dss_line_iter_runs(eeg_with_line_noise):
    """Run dss_line_iter zapline without error, preserve shape."""
    raw_clean = compute_zapline(
        eeg_with_line_noise, noise_freqs=50.0, method="dss_line_iter"
    )
    assert raw_clean.get_data().shape == eeg_with_line_noise.get_data().shape


def test_compute_zapline_dss_line_reduces_noise(eeg_with_line_noise):
    """Power at 50 Hz should decrease after ZapLine."""
    sfreq = eeg_with_line_noise.info["sfreq"]
    data_before = eeg_with_line_noise.get_data(picks=[0])
    raw_clean = compute_zapline(
        eeg_with_line_noise, noise_freqs=50.0, method="dss_line"
    )
    data_after = raw_clean.get_data(picks=[0])

    freqs = np.fft.rfftfreq(data_before.shape[1], d=1.0 / sfreq)
    idx_50 = np.argmin(np.abs(freqs - 50.0))

    psd_before = np.abs(np.fft.rfft(data_before[0])) ** 2
    psd_after = np.abs(np.fft.rfft(data_after[0])) ** 2

    assert psd_after[idx_50] < psd_before[idx_50]


def test_compute_zapline_europe_preset(eeg_with_line_noise):
    """Verify compute_zapline accepts the europe noise_freqs preset."""
    raw_clean = compute_zapline(
        eeg_with_line_noise, noise_freqs="europe", method="dss_line"
    )
    assert raw_clean.get_data().shape == eeg_with_line_noise.get_data().shape


def test_compute_zapline_usa_preset(eeg_with_line_noise):
    """Verify compute_zapline accepts the usa noise_freqs preset."""
    # usa has 60 Hz - above half of 500 Hz sfreq is fine, below Nyquist
    raw_clean = compute_zapline(
        eeg_with_line_noise, noise_freqs="usa", method="dss_line"
    )
    assert raw_clean is not None


def test_compute_zapline_unknown_preset_raises(eeg_with_line_noise):
    """Raise ValueError for unrecognised noise_freqs preset."""
    with pytest.raises(ValueError, match="Unknown noise_freqs preset"):
        compute_zapline(eeg_with_line_noise, noise_freqs="asia")


def test_compute_zapline_unknown_method_raises(eeg_with_line_noise):
    """Raise ValueError for unrecognised zapline method."""
    with pytest.raises(ValueError, match="Unknown zapline method"):
        compute_zapline(eeg_with_line_noise, noise_freqs=50.0, method="bogus_method")


def test_compute_zapline_above_nyquist_skipped(eeg_with_line_noise):
    """Skip frequencies above Nyquist, return raw unchanged."""
    sfreq = eeg_with_line_noise.info["sfreq"]
    raw_clean = compute_zapline(
        eeg_with_line_noise, noise_freqs=sfreq, method="dss_line"
    )
    np.testing.assert_array_equal(raw_clean.get_data(), eeg_with_line_noise.get_data())


def test_compute_zapline_none_freq_raises_for_dss(eeg_with_line_noise):
    """Raise ValueError when noise_freqs is None."""
    with pytest.raises(ValueError, match="noise_freqs cannot be None"):
        compute_zapline(eeg_with_line_noise, noise_freqs=None, method="dss_line")


# %% detect_bad_by_line_noise


def test_detect_bad_by_line_noise_returns_list(eeg_with_line_noise):
    """Verify detect_bad_by_line_noise returns a list."""
    bads = detect_bad_by_line_noise(eeg_with_line_noise, noise_freqs=[50.0])
    assert isinstance(bads, list)


def test_detect_bad_by_line_noise_detects_noisy_channel():
    """Flag channel with 1000x stronger 50 Hz noise."""
    rng = np.random.default_rng(0)
    sfreq = 500.0
    n_ch = 32
    n_times = int(sfreq * 20)
    t = np.arange(n_times) / sfreq
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    # mild background noise on all channels
    data = rng.standard_normal((n_ch, n_times)) * 1e-6
    # inject strong 50 Hz noise on channel 0 only
    data[0] += np.sin(2 * np.pi * 50 * t) * 1e-4
    raw = mne.io.RawArray(data, info, verbose=False)
    bads = detect_bad_by_line_noise(raw, noise_freqs=[50.0], z_thresh=3.0)
    assert "EEG000" in bads


def test_detect_bad_by_line_noise_empty_when_no_eeg(rng):
    """Return empty list when no EEG channels present."""
    info = mne.create_info(["A", "B"], sfreq=250.0, ch_types="misc")
    data = rng.standard_normal((2, 500)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    bads = detect_bad_by_line_noise(raw, noise_freqs=[50.0])
    assert bads == []


# %% compute_mi_reduction


def test_compute_mi_reduction_returns_expected_keys(tiny_raw):
    """Return dict with all four expected keys."""
    result = compute_mi_reduction(tiny_raw, tiny_raw)
    assert set(result.keys()) == {
        "mi_before",
        "mi_after",
        "mi_reduction",
        "mi_reduction_pct",
    }


def test_compute_mi_reduction_identical_raws(tiny_raw):
    """Before equals after, reduction should be zero."""
    result = compute_mi_reduction(tiny_raw, tiny_raw)
    assert result["mi_reduction"] == pytest.approx(0.0, abs=1e-9)


def test_compute_mi_reduction_after_less_than_before(rng):
    """Removing shared variance should reduce MI."""
    sfreq = 250.0
    n_times = int(sfreq * 10)
    n_ch = 16
    noise = rng.standard_normal((1, n_times))  # shared across all channels
    signal = rng.standard_normal((n_ch, n_times)) * 1e-6

    # before: high MI (shared noise)
    data_before = signal + noise * 1e-5
    # after: no shared noise
    data_after = signal.copy()

    info = mne.create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq, "eeg")
    raw_before = mne.io.RawArray(data_before, info, verbose=False)
    raw_after = mne.io.RawArray(data_after, info, verbose=False)

    result = compute_mi_reduction(raw_before, raw_after)
    assert result["mi_reduction"] > 0


# %% compute_asr


def test_compute_asr_preserves_shape(tiny_raw):
    """Preserve Raw data shape after ASR."""
    raw_asr = compute_asr(tiny_raw)
    assert raw_asr.get_data().shape == tiny_raw.get_data().shape


def test_compute_asr_eeg_channels_modified(rng):
    """Modify only EEG channels, leave misc unchanged."""
    sfreq = 250.0
    n_times = int(sfreq * 10)
    ch_names = [f"EEG{i:03d}" for i in range(8)] + ["Misc0", "Misc1"]
    ch_types = ["eeg"] * 8 + ["misc"] * 2
    info = mne.create_info(ch_names, sfreq, ch_types)
    data = rng.standard_normal((10, n_times)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)

    raw_asr = compute_asr(raw)

    # misc channels unchanged
    np.testing.assert_array_equal(
        raw_asr.get_data(picks="misc"),
        raw.get_data(picks="misc"),
    )


# %% build_sys_info


def test_build_sys_info_contains_key_packages():
    """Include MNE and installed-packages section in output."""
    info_str = build_sys_info()
    assert "mne" in info_str.lower()
    assert "Installed packages" in info_str


def test_build_sys_info_includes_source_data(tmp_path):
    """Include source data file paths in output."""
    f = tmp_path / "sub-01.xdf"
    f.touch()
    info_str = build_sys_info(source_data=[f])
    assert str(f) in info_str
