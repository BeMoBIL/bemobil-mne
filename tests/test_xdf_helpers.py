"""Tests for bpn_analysis.io.xdf private helpers."""

# %% Imports

import logging

import pytest

from bpn_analysis.io.xdf import (
    _FIF_CH_ABBREVIATIONS,
    _FIF_MAX_CH_LEN,
    _match_stream,
    _shorten_fif_ch_names,
    _unit_scale,
)

# %% _shorten_fif_ch_names


def test_shorten_passthrough_short_names():
    """Names at or below 15 chars must pass through unchanged."""
    names = ["Cz", "EEG001", "pupil_left"]
    assert _shorten_fif_ch_names(names) == names


def test_shorten_uses_abbreviation_table():
    """Known Pupil Labs names > 15 chars must be replaced by their MNE short form."""
    # "EyeballCenterX-0" is 16 chars -> triggers abbreviation lookup
    names = ["EyeballCenterX-0", "EyeballCenterX-1"]
    result = _shorten_fif_ch_names(names)
    assert result == ["eyeball_x_l", "eyeball_x_r"]


def test_shorten_all_known_eye_tracker_channels():
    """Every entry in _FIF_CH_ABBREVIATIONS that exceeds 15 chars must map correctly."""
    long_names = [n for n in _FIF_CH_ABBREVIATIONS if len(n) > _FIF_MAX_CH_LEN]
    assert long_names, "Expected at least some names > 15 chars in the table"
    result = _shorten_fif_ch_names(long_names)
    for original, shortened in zip(long_names, result):
        assert len(shortened) <= _FIF_MAX_CH_LEN, (
            f"{original!r} -> {shortened!r} is {len(shortened)} chars"
        )
        assert shortened == _FIF_CH_ABBREVIATIONS[original]


def test_shorten_fallback_truncates_unknown():
    """Unknown names longer than 15 chars are truncated to exactly 15."""
    name = "ThisIsAVeryLongChannelName"
    result = _shorten_fif_ch_names([name])
    assert len(result[0]) == _FIF_MAX_CH_LEN
    assert result[0] == name[:_FIF_MAX_CH_LEN]


def test_shorten_deduplicates_on_truncation():
    """If two names truncate to the same prefix, the second gets a unique suffix."""
    # Create two names that share the first 15 characters
    base = "A" * _FIF_MAX_CH_LEN
    name1 = base + "X"
    name2 = base + "Y"
    result = _shorten_fif_ch_names([name1, name2])
    assert result[0] != result[1], "Truncated names must be disambiguated"
    assert all(len(r) <= _FIF_MAX_CH_LEN for r in result)


def test_shorten_output_all_within_limit():
    """All output names must satisfy the FIF limit regardless of input."""
    mixed = [
        "short",
        "EyelidAngleTopLeft",        # in table
        "UnknownVeryLongChannelName", # fallback
        "AnotherLongNameThatWontFit", # fallback (may collide)
    ]
    result = _shorten_fif_ch_names(mixed)
    for name in result:
        assert len(name) <= _FIF_MAX_CH_LEN


def test_shorten_output_unique():
    """Returned names must be globally unique."""
    names = ["EyelidAngleTopLeft", "AnotherLongChannelNameThatTruncatesToSamePrefix"]
    result = _shorten_fif_ch_names(names)
    assert len(result) == len(set(result))


def test_shorten_logs_rename(caplog):
    """A rename must be logged at INFO level so it is auditable."""
    with caplog.at_level(logging.INFO):
        _shorten_fif_ch_names(["EyeballCenterX-0"])  # 16 chars -> triggers rename
    assert "EyeballCenterX-0" in caplog.text


# %% _unit_scale


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("microvolt", 1e-6),
        ("microvolts", 1e-6),
        ("µV", 1e-6),
        ("uV", 1e-6),
        ("V", 1.0),
        ("volt", 1.0),
        ("Volt", 1.0),
        ("NA", 1.0),
        ("", 1.0),
    ],
)
def test_unit_scale_known_units(unit, expected):
    assert _unit_scale(unit) == pytest.approx(expected)


def test_unit_scale_unknown_unit_warns(caplog):
    """Unrecognised units emit a warning but return 1.0 (identity scaling)."""
    with caplog.at_level(logging.WARNING):
        scale = _unit_scale("furlong", ch_name="ch0")
    assert scale == pytest.approx(1.0)
    assert "furlong" in caplog.text


def test_unit_scale_case_insensitive():
    assert _unit_scale("MICROVOLT") == pytest.approx(1e-6)
    assert _unit_scale("MicroVolt") == pytest.approx(1e-6)


# %% _match_stream


def _make_stream(name="EEG", stream_type="EEG", source_id="device01"):
    return {
        "info": {
            "name": [name],
            "type": [stream_type],
            "source_id": [source_id],
            "channel_count": ["1"],
        },
        "time_series": [],
        "time_stamps": [],
    }


def test_match_stream_by_name():
    streams = [_make_stream("EEG"), _make_stream("EMG")]
    result = _match_stream(streams, name="EEG")
    assert result["info"]["name"][0] == "EEG"


def test_match_stream_by_type():
    streams = [_make_stream("A", stream_type="EEG"), _make_stream("B", stream_type="EMG")]
    result = _match_stream(streams, stream_type="EMG")
    assert result["info"]["name"][0] == "B"


def test_match_stream_by_source_id():
    streams = [
        _make_stream(source_id="device01"),
        _make_stream(source_id="device02"),
    ]
    result = _match_stream(streams, source_id="device02")
    assert result["info"]["source_id"][0] == "device02"


def test_match_stream_no_match_returns_none():
    streams = [_make_stream("EEG")]
    result = _match_stream(streams, name="NonExistent")
    assert result is None


def test_match_stream_allow_multiple():
    streams = [_make_stream("EEG", stream_type="EEG")] * 3
    result = _match_stream(streams, stream_type="EEG", allow_multiple=True)
    assert isinstance(result, list)
    assert len(result) == 3


def test_match_stream_allow_multiple_no_match_returns_empty_list():
    streams = [_make_stream("EEG")]
    result = _match_stream(streams, name="Ghost", allow_multiple=True)
    assert result == []


def test_match_stream_multiple_warns_and_returns_first(caplog):
    """When allow_multiple=False and multiple streams match, first is returned with a warning."""
    streams = [_make_stream("EEG")] * 2
    with caplog.at_level(logging.WARNING):
        result = _match_stream(streams, name="EEG")
    assert result is not None
    assert "Multiple" in caplog.text
