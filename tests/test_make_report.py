"""Tests for bemobil_mne.preproc.make_report."""

# %% Imports

import matplotlib
import matplotlib.pyplot as plt
import mne
import pytest

from bemobil_mne.preproc import make_report

# %% Helpers


def _raw_with_events(rng):
    montage = mne.channels.make_standard_montage("easycap-M1")
    info = mne.create_info(montage.ch_names[:8], 250.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((8, 250 * 20)) * 1e-6, info)
    raw.set_montage(montage)
    raw.set_annotations(mne.Annotations([2, 5, 8, 11], 0, ["a", "a", "a", "a"]))
    return raw


def _report(raw, **kwargs):
    ica = mne.preprocessing.ICA(method="picard")  # unfitted: ICA part skipped
    return make_report(
        raw_pre_ica=raw,
        raw_clean=raw,
        ica=ica,
        ic_labels={},
        dipoles=[],
        residuals=[],
        trans=None,
        bad_ch_dict={"all_bads": []},
        **kwargs,
    )


# %% Tests


def test_make_report_leaves_matplotlib_and_logging_alone(rng):
    """Backend, open user figures and the MNE log level are unchanged."""
    backend = matplotlib.get_backend()
    user_fig = plt.figure()
    mne.set_log_level("INFO")
    try:
        _report(_raw_with_events(rng))
        assert matplotlib.get_backend() == backend
        assert plt.fignum_exists(user_fig.number)
        assert mne.utils.logger.level == 20  # INFO
    finally:
        plt.close(user_fig)


def test_make_report_skips_missing_event_ids(rng):
    """Conditions absent from the data are skipped, not fatal."""
    report = _report(_raw_with_events(rng), event_id={"a": 1, "missing": 2})
    html = "".join(report.html)
    assert "Not found in the data (skipped)" in html
    assert "missing" in html


def test_make_report_escapes_pre_hook_source(rng):
    """Pre-hook source code is HTML-escaped."""
    src = "def hook(raw):\n    return raw if 1 < 2 else None  # <b>x</b>\n"
    report = _report(_raw_with_events(rng), pre_hook_source=src)
    html = "".join(report.html)
    assert "1 &lt; 2" in html
    assert "<b>x</b>" not in html


@pytest.mark.parametrize("event_id", [None, {"a": 1}])
def test_make_report_runs(rng, event_id):
    """Build a report with and without an event map."""
    assert isinstance(_report(_raw_with_events(rng), event_id=event_id), mne.Report)
