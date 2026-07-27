"""BIDS export utilities for bpn_analysis.

Thin wrappers around :mod:`mne_bids` that apply project conventions and
optionally record export parameters in the raw object's provenance descriptor.

Requires the ``mne_bids`` package (already a declared dependency of
``BPN-analysis``).
"""

# %% Imports

from __future__ import annotations

import logging
from pathlib import Path

import mne_bids

# %% Settings & Constants

LOGGER = logging.getLogger(__name__)

# %% Functions


def export_to_bids(
    raw,
    bids_root,
    subject,
    session=None,
    task="task",
    run=None,
    acquisition=None,
    events=None,
    event_id=None,
    line_freq=50.0,
    overwrite=False,
    verbose=True,
    extra_params=None,
):
    """Export an MNE Raw object to a BIDS-compatible directory tree.

    Wraps :func:`mne_bids.write_raw_bids` with sensible defaults for EEG
    data and records the BIDS path in the raw object's provenance descriptor
    (when one exists).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG recording.  Must have a montage set and ``info['sfreq']``
        populated.
    bids_root : str | Path
        Root of the BIDS dataset (will be created if it does not exist).
    subject : str
        BIDS subject label (without the ``sub-`` prefix, e.g. ``"01"``).
    session : str | None
        BIDS session label (without ``ses-``).  ``None`` omits the session
        directory level.
    task : str
        BIDS task label.  Defaults to ``"task"``.
    run : str | None
        BIDS run label (without ``run-``).  ``None`` when there is a single
        run.
    acquisition : str | None
        BIDS acquisition label (without ``acq-``).
    events : ndarray, shape (n_events, 3) | None
        MNE-style event array (sample, prev_id, event_id).  When ``None``,
        events are derived from ``raw.annotations`` automatically by
        :func:`mne_bids.write_raw_bids`.
    event_id : dict | None
        Mapping of event descriptions to integer codes, e.g.
        ``{"left_hand": 1, "right_hand": 2}``.
    line_freq : float
        Power-line frequency (Hz).  Used to populate the ``PowerLineFrequency``
        BIDS field.  Defaults to 50 Hz (Europe / most of Asia).
    overwrite : bool
        Whether to overwrite existing BIDS files for this entity.
    verbose : bool | str | int
        Verbosity level forwarded to :func:`mne_bids.write_raw_bids`.
    extra_params : dict | None
        Extra keyword arguments forwarded verbatim to
        :func:`mne_bids.write_raw_bids`.

    Returns
    -------
    bids_path : mne_bids.BIDSPath
        The :class:`mne_bids.BIDSPath` object describing the written file.
    """
    bids_root = Path(bids_root)

    bids_path = mne_bids.BIDSPath(
        subject=str(subject),
        session=str(session) if session is not None else None,
        task=str(task),
        run=str(run) if run is not None else None,
        acquisition=str(acquisition) if acquisition is not None else None,
        root=bids_root,
        datatype="eeg",
    )

    # Set power-line frequency in info (required by BIDS validator)
    raw = raw.copy()
    raw.info["line_freq"] = float(line_freq)

    write_kwargs: dict = dict(
        raw=raw,
        bids_path=bids_path,
        overwrite=overwrite,
        verbose=verbose,
    )
    if events is not None:
        write_kwargs["events"] = events
    if event_id is not None:
        write_kwargs["event_id"] = event_id
    if extra_params:
        write_kwargs.update(extra_params)

    bids_path_out = mne_bids.write_raw_bids(**write_kwargs)
    LOGGER.info(f"BIDS export complete: {bids_path_out.fpath}")

    # Record in provenance if descriptor exists
    try:
        from bpn_analysis.preproc.utils import append_desc, get_descriptor

        if get_descriptor(raw) is not None:
            append_desc(
                raw,
                name="bids_export",
                bids_path=str(bids_path_out.fpath),
                subject=subject,
                session=session,
                task=task,
                run=run,
            )
    except Exception:
        pass  # provenance recording is best-effort

    return bids_path_out


def make_bids_dataset_description(
    bids_root,
    name,
    authors=None,
    version="1.0.0",
    license_text=None,
    acknowledgements=None,
    funding=None,
    doi=None,
):
    """Write a minimal BIDS ``dataset_description.json``.

    Calls :func:`mne_bids.make_dataset_description`.  Idempotent: safe to
    call multiple times.

    Parameters
    ----------
    bids_root : str | Path
        Root of the BIDS dataset.
    name : str
        Human-readable dataset name.
    authors : list of str | None
        Author names.
    version : str
        Dataset version string.
    license_text : str | None
        SPDX license identifier or free-text.
    acknowledgements : str | None
        Acknowledgements text.
    funding : list of str | None
        Funding source strings.
    doi : str | None
        Dataset DOI.
    """
    mne_bids.make_dataset_description(
        path=str(bids_root),
        name=name,
        authors=authors or [],
        version=version,
        license=license_text or "",
        acknowledgements=acknowledgements or "",
        funding=funding or [],
        doi=doi or "",
        overwrite=True,
    )
    LOGGER.info(f"BIDS dataset_description.json written at {bids_root}")


def batch_export_to_bids(
    subject_runs,
    bids_root,
    task="task",
    line_freq=50.0,
    overwrite=False,
    verbose=False,
):
    """Export multiple subjects / runs to BIDS in a single call.

    Parameters
    ----------
    subject_runs : list of dict
        Each dict must contain at least ``{"raw": <raw>, "subject": "01"}``.
        Optional keys: ``session``, ``run``, ``acquisition``, ``events``,
        ``event_id``.
    bids_root : str | Path
        BIDS root directory.
    task : str
        Task label shared across all runs.
    line_freq : float
        Power-line frequency.
    overwrite : bool
        Overwrite existing BIDS files.
    verbose : bool | str | int
        Verbosity for :func:`mne_bids.write_raw_bids`.

    Returns
    -------
    bids_paths : list of mne_bids.BIDSPath
        One path per successfully written run.
    """
    results = []
    for entry in subject_runs:
        raw = entry["raw"]
        subject = str(entry["subject"])
        session = entry.get("session")
        run = entry.get("run")
        acquisition = entry.get("acquisition")
        events = entry.get("events")
        event_id = entry.get("event_id")

        try:
            bp = export_to_bids(
                raw=raw,
                bids_root=bids_root,
                subject=subject,
                session=session,
                task=task,
                run=run,
                acquisition=acquisition,
                events=events,
                event_id=event_id,
                line_freq=line_freq,
                overwrite=overwrite,
                verbose=verbose,
            )
            results.append(bp)
        except Exception as exc:
            LOGGER.error(f"BIDS export failed for sub-{subject}: {exc}")

    return results
