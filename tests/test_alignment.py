"""Tests for pybemobil.io.alignment."""

# %% Imports

import numpy as np
import pytest

from pybemobil.io.alignment import align_stream_to_timestamps

# %% Tests


def test_long_gap_kept_in_any_channel():
    """A long NaN gap is preserved even when it is not in channel 0."""
    ts = np.arange(0, 10, 0.01)
    data = np.c_[np.sin(ts), np.cos(ts)]
    data[300:500, 1] = np.nan  # 2 s gap in channel 1 only
    out = align_stream_to_timestamps(data, ts, ts, method="linear", max_nan_gap_s=0.5)
    assert np.isnan(out[300:500, 1]).all()
    assert not np.isnan(out[:, 0]).any()  # channel 0 is untouched


def test_short_gap_is_filled():
    """Gaps shorter than max_nan_gap_s are bridged."""
    ts = np.arange(0, 10, 0.01)
    data = np.sin(ts)
    data[300:305] = np.nan
    out = align_stream_to_timestamps(data, ts, ts, max_nan_gap_s=0.5)
    assert not np.isnan(out).any()


def test_sinc_with_nan_at_stream_edges():
    """Leading NaN must not turn the whole sinc-aligned channel into NaN."""
    ts = np.arange(0, 10, 1 / 1000)
    data = np.sin(2 * np.pi * 3 * ts)
    data[:50] = np.nan
    tgt = np.arange(0, 10, 1 / 250)
    out = align_stream_to_timestamps(data, ts, tgt, method="sinc")[:, 0]
    assert np.isnan(out).mean() < 0.01
    ok = tgt > 0.5
    np.testing.assert_allclose(out[ok], np.sin(2 * np.pi * 3 * tgt[ok]), atol=0.02)


@pytest.mark.parametrize("src_sfreq", [500, 2000, 5000])
def test_sinc_downsampling_is_stable(src_sfreq):
    """The anti-aliasing filter stays stable for large decimation ratios."""
    ts = np.arange(0, 10, 1 / src_sfreq)
    data = np.sin(2 * np.pi * 3 * ts) + np.sin(2 * np.pi * 200 * ts)
    tgt = np.arange(1, 9, 1 / 100)
    out = align_stream_to_timestamps(data, ts, tgt, method="sinc")[:, 0]
    # 3 Hz passes, 200 Hz (above the 100 Hz target's Nyquist) is removed
    np.testing.assert_allclose(out, np.sin(2 * np.pi * 3 * tgt), atol=0.05)
