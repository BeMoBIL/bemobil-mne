"""Tests for bemobil_mne.viz.plotting."""

# %% Imports

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bemobil_mne.viz import (  # noqa: E402
    plot_ERP,
    plot_PSD,
    plot_psd_topomaps,
    plot_TFR,
    plot_tfr_topomaps,
    plot_various_ERPs,
)
from bemobil_mne.viz.plotting import _validate_interval  # noqa: E402

# %% Fixtures


@pytest.fixture
def raw(rng):
    """8-channel EEG (100 Hz, 40 s) with 'a'/'b' events every 2 s."""
    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]
    info = mne.create_info(ch_names, 100.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((8, 4000)) * 1e-6, info, verbose=False)
    raw.set_montage(montage)
    onsets = np.arange(2, 38, 2.0)
    raw.set_annotations(
        mne.Annotations(onsets, 0, ["a" if i % 2 else "b" for i in range(len(onsets))])
    )
    return raw


@pytest.fixture
def epochs(raw):
    """Epochs from -0.5 to 1.5 s around all events."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return mne.Epochs(
        raw, events, event_id, tmin=-0.5, tmax=1.5, preload=True, verbose=False
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


FREQS = np.arange(4.0, 20.0, 2.0)

# %% Tests


def test_plot_erp_accepts_evoked(epochs):
    """plot_ERP works with a dict of Evoked, as documented."""
    fig = plot_ERP({"a": epochs["a"].average()}, picks=["Cz"])
    assert fig.axes


def test_plot_psd_accepts_evoked(epochs):
    """plot_PSD works with a dict of Evoked."""
    fig = plot_PSD({"a": epochs["a"].average()}, 1, 30, picks=["Cz"], ci=False)
    assert fig.axes[0].get_title() == "Cz"


def test_plot_psd_without_ci_in_db(epochs):
    """ci=False no longer raises a KeyError when dB=True."""
    plot_PSD({"a": epochs["a"]}, 1, 30, ci=False, dB=True)


def test_plot_psd_picks_match_titles_and_data(epochs):
    """Each panel shows the PSD of the channel in its title."""
    fig = plot_PSD({"a": epochs["a"]}, 1, 30, picks=["Oz"], ci=False, dB=False)
    ax = fig.axes[0]
    assert ax.get_title() == "Oz"
    expected = epochs["a"].compute_psd(
        fmin=1, fmax=30, picks=["Oz"], method="multitaper", verbose=False
    )
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), expected.get_data().mean(axis=0)[0], rtol=1e-6
    )


def test_plot_tfr_accepts_epochs_read_from_disk(epochs, tmp_path):
    """Epochs loaded with mne.read_epochs (EpochsFIF) are accepted."""
    fname = tmp_path / "x-epo.fif"
    epochs.save(fname, verbose=False)
    loaded = mne.read_epochs(fname, verbose=False)
    figs = plot_TFR({"a": loaded["a"]}, freqs=FREQS, picks=["Cz"])
    assert set(figs) == {"a"}


def test_plot_tfr_does_not_modify_input(epochs):
    """Baseline correction and cropping work on a copy of the TFR."""
    tfr = epochs["a"].compute_tfr(
        "morlet", freqs=FREQS, n_cycles=2, average=True, verbose=False
    )
    data, n_times = tfr.data.copy(), len(tfr.times)
    plot_TFR(
        {"a": tfr}, freqs=FREQS, tmin=0, tmax=0.4, baseline=(-0.2, 0), picks=["Cz"]
    )
    assert len(tfr.times) == n_times
    np.testing.assert_array_equal(tfr.data, data)


def test_plot_tfr_topomaps_with_vlim(epochs):
    """A user-given vlim works and the caller's plot_kwargs are not modified."""
    tfr = epochs["a"].compute_tfr(
        "morlet", freqs=FREQS, n_cycles=2, average=True, verbose=False
    )
    plot_kwargs = {"vlim": (-1, 1)}
    figs = plot_tfr_topomaps(
        {"a": tfr}, bands={"t": (4, 8)}, times=[(0, 0.2)], plot_kwargs=plot_kwargs
    )
    assert set(figs) == {"a"}
    assert plot_kwargs == {"vlim": (-1, 1)}


def _colorbar_limits(fig, label):
    cbars = [ax for ax in fig.axes if ax.get_ylabel() == label]
    assert cbars, f"no colorbar labelled {label!r}"
    return cbars[0].get_ylim()


def test_plot_psd_topomaps_absolute_is_in_microvolts(epochs):
    """Absolute band power is scaled to µV²/Hz, as labelled."""
    fig = plot_psd_topomaps({"a": epochs["a"]}, unit="absolute", bands={"a": (8, 12)})
    lo, hi = _colorbar_limits(fig, "µV²/Hz")
    assert 1e-4 < hi < 1e4  # V²/Hz would be ~1e-12


@pytest.mark.parametrize("unit", ["relative", (1, 45)])
def test_plot_psd_topomaps_relative_is_a_fraction(epochs, unit):
    """Relative power is a fraction of the reference band; tuples normalise too."""
    bands = {"low": (1, 20), "high": (20, 45)}
    fig = plot_psd_topomaps({"a": epochs["a"]}, unit=unit, bands=bands)
    highs = [ax.get_ylim()[1] for ax in fig.axes if ax.get_ylabel() == "Relative power"]
    assert len(highs) == 2
    assert all(0 < h <= 1 for h in highs)


def test_validate_interval():
    """Numpy numbers are accepted and non-numbers rejected."""
    assert _validate_interval((np.int64(0), 1), "x") == (0.0, 1.0)
    with pytest.raises(TypeError, match="must be numbers"):
        _validate_interval((0, "a"), "x")
    with pytest.raises(ValueError, match="smaller"):
        _validate_interval((1, 0), "x")


def test_plot_various_erps_does_not_modify_inputs(raw):
    """Resampling and subtraction leave the caller's raw and event_id alone."""
    event_id = {"a": 1, "b": 2}
    sfreq = raw.info["sfreq"]
    plot_various_ERPs(
        raw,
        event_id=event_id,
        picks=["Cz"],
        sfreq=50.0,
        subtraction_event="b",
        subtraction_targets=["a"],
        epochs_kwargs={"tmin": -0.2, "tmax": 0.5},
    )
    assert raw.info["sfreq"] == sfreq
    assert event_id == {"a": 1, "b": 2}
