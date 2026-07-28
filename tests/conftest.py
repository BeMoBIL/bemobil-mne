"""Shared pytest fixtures and configuration for bemobil_mne tests."""

# %% Imports

import subprocess

import mne
import numpy as np
import pytest

# %% Fixtures


@pytest.fixture(scope="session")
def rng():
    """Fix random state for reproducible synthetic data."""
    return np.random.default_rng(42)


@pytest.fixture
def tiny_raw(rng):
    """Minimal synthetic 16-channel EEG Raw (250 Hz, 10 s).

    Fast to create - use this in unit tests that only need an EEG Raw
    and do not depend on realistic channel names or montage positions.
    """
    sfreq = 250.0
    n_times = int(sfreq * 10)
    ch_names = [f"EEG{i:03d}" for i in range(16)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.standard_normal((16, n_times)) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture
def tiny_raw_with_montage(rng):
    """16-channel EEG Raw with standard_1020 montage (250 Hz, 10 s).

    Use when channel positions are required (e.g. BIDS export, dipole fitting).
    """
    sfreq = 250.0
    n_times = int(sfreq * 10)
    montage = mne.channels.make_standard_montage("standard_1020")
    # pick first 16 channels that exist in the montage
    ch_names = montage.ch_names[:16]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.standard_normal((16, n_times)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")
    return raw


@pytest.fixture
def motion_raw(rng):
    """Synthetic Raw with two rigid bodies.

    ``head``: quaternion + position channels.
    ``hand``: Euler + position channels.

    Used in motion processing tests.
    """
    sfreq = 100.0
    n_times = int(sfreq * 5)

    # head: quat_x/y/z/w + pos_x/y/z
    q = rng.standard_normal((4, n_times))
    q /= np.linalg.norm(q, axis=0, keepdims=True)
    pos_head = rng.standard_normal((3, n_times)) * 0.05

    # hand: eul_x/y/z + pos_x/y/z
    eul_hand = rng.standard_normal((3, n_times)) * 0.1
    pos_hand = rng.standard_normal((3, n_times)) * 0.05

    ch_names = [
        "head_quat_x",
        "head_quat_y",
        "head_quat_z",
        "head_quat_w",
        "head_pos_x",
        "head_pos_y",
        "head_pos_z",
        "hand_eul_x",
        "hand_eul_y",
        "hand_eul_z",
        "hand_pos_x",
        "hand_pos_y",
        "hand_pos_z",
    ]
    data = np.vstack([q, pos_head, eul_hand, pos_hand])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture(scope="session")
def sample_raw():
    """PhysioNet EEG motor-imagery run 6 (64 ch, ~160 Hz).

    Downloaded once per test session.  Marked as ``slowtest`` where used,
    so regular ``pytest`` skips it unless ``--runslow`` is passed.
    """
    subjects = [1]
    runs = [6]
    raws = mne.datasets.eegbci.load_data(subjects[0], runs, verbose=False)
    raw = mne.io.concatenate_raws(
        [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raws]
    )
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    return raw


# %% pytest hooks


def pytest_addoption(parser):
    """Register custom CLI flags."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slowtest",
    )


def pytest_configure(config):
    """Register project-specific markers."""
    config.addinivalue_line(
        "markers",
        "slowtest: mark test as slow (skipped unless --runslow is passed)",
    )
    config.addinivalue_line(
        "markers",
        "requires_data: mark test as requiring external data files",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip slow tests unless --runslow flag is present."""
    skip_slow = pytest.mark.skip(reason="pass --runslow to run slow tests")
    for item in items:
        if "slowtest" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)


def pytest_sessionfinish(session, exitstatus):
    """Print a brief coverage summary after the test run."""
    result = subprocess.run(
        ["python", "-m", "coverage", "report", "--skip-empty"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print("\n\nCoverage summary:\n")
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
