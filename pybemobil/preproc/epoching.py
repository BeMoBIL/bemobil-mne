"""Epoch preparation: event remapping, filtering, and epoching."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mne
import numpy as np

LOGGER = logging.getLogger(__name__)


def get_stimulus_rename_map(
    descriptions: list[str],
) -> tuple[dict[str, str], dict[str, int]]:
    """Build an event rename map from ``trialStart`` annotation descriptions.

    Parses ``condition:<name>`` and ``stimulus:<name>`` fields out of each
    ``trialStart`` annotation and produces a mapping from the raw description
    string to a ``"condition/stimulus"`` label, plus a label-to-integer ID
    lookup.

    Parameters
    ----------
    descriptions : list of str
        Annotation description strings from ``raw.annotations.description``.

    Returns
    -------
    rename_map : dict[str, str]
        Maps each raw ``trialStart`` description to its ``"condition/stimulus"``
        label.
    label_to_id : dict[str, int]
        Maps each unique ``"condition/stimulus"`` label to an integer event ID
        (1-indexed, sorted alphabetically).
    """
    unique_labels: set[str] = set()

    for desc in descriptions:
        if not desc.startswith("trialStart"):
            continue
        condition_match = re.search(r"condition:(\w+)", desc)
        stimulus_match = re.search(r"stimulus:(\w+)", desc)
        if condition_match and stimulus_match:
            unique_labels.add(f"{condition_match.group(1)}/{stimulus_match.group(1)}")

    label_to_id = {label: i + 1 for i, label in enumerate(sorted(unique_labels))}

    rename_map: dict[str, str] = {}
    for desc in descriptions:
        if not desc.startswith("trialStart"):
            continue
        condition_match = re.search(r"condition:(\w+)", desc)
        stimulus_match = re.search(r"stimulus:(\w+)", desc)
        if condition_match and stimulus_match:
            label = f"{condition_match.group(1)}/{stimulus_match.group(1)}"
            rename_map[desc] = label

    LOGGER.info(
        "Discovered event types: "
        + ", ".join(f"{id_}: {label}" for label, id_ in label_to_id.items())
    )

    return rename_map, label_to_id


class EpochPreparer:
    """Prepare condition-specific epochs from a cleaned raw recording.

    Handles event remapping (from raw ``trialStart`` annotations to
    condition/stimulus labels and then to user-supplied category labels),
    optional latency correction, bandpass filtering, and epoching.

    Parameters
    ----------
    remaps : dict[str, dict[str, str]]
        Mapping of condition name → {original_label: target_label}.
        Only annotations whose remapped value appears in the condition's remap
        dict values are retained for epoching.
    epoch_tlimes : tuple of float
        ``(tmin, tmax)`` in seconds for epochs relative to the event onset.
    baseline : tuple of float | None
        Baseline window ``(tmin, tmax)`` passed to ``mne.Epochs``.
    bandpass_erp : tuple of (float | None, float | None)
        ``(l_freq, h_freq)`` bandpass applied before epoching.
    tshift : float
        Stimulus onset delay in seconds.  Event annotation onsets are
        shifted forward by this amount so that ``t=0`` aligns with actual
        stimulus presentation; ``BAD_*`` / ``EDGE`` annotations are not
        shifted.  Set to ``0`` to disable.

    Notes
    -----
    ``BAD_*`` annotations (e.g. breaks or missing auxiliary data) are kept,
    so epochs overlapping them are rejected by :class:`mne.Epochs`.
    """

    def __init__(
        self,
        remaps: dict[str, dict[str, str]],
        *,
        epoch_tlimes: tuple[float, float] = (-0.2, 0.8),
        baseline: tuple[float | None, float | None] = (-0.2, 0),
        bandpass_erp: tuple[float | None, float | None] = (None, 20.0),
        tshift: float = 0.060,
    ):
        self.remaps = remaps
        self.epoch_tlimes = epoch_tlimes
        self.baseline = baseline
        self.bandpass_erp = bandpass_erp
        self.tshift = tshift

    def run(
        self,
        raw_clean: mne.io.BaseRaw,
        fname_out: str | Path,
        cond: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, mne.Epochs]:
        """Run event remapping, filtering, and epoching for one condition.

        Parameters
        ----------
        raw_clean : mne.io.BaseRaw
            ICA-cleaned raw recording that still carries all annotations.
        fname_out : str | Path
            Output stem; ``_epo.fif.gz`` is appended automatically (a
            trailing ``.fif`` / ``.fif.gz`` is removed first).
        cond : str
            Condition key that must exist in ``self.remaps``.
        overwrite : bool
            Overwrite existing epoch file.

        Returns
        -------
        epochs_dict : dict[str, mne.Epochs]
            ``{label: epochs}`` for each target label in the condition remap
            that occurs in the data (missing labels are logged and skipped).
        """
        fname_out = Path(fname_out)
        for ext in (".fif.gz", ".fif"):
            if fname_out.name.endswith(ext):
                fname_out = fname_out.with_name(fname_out.name[: -len(ext)])
                break
        remap = self.remaps[cond]
        targets = list(dict.fromkeys(remap.values()))

        rename_map, _ = get_stimulus_rename_map(raw_clean.annotations.description)

        raw_remap = raw_clean.copy().load_data()
        raw_remap.annotations.rename(rename_map)
        present = set(raw_remap.annotations.description)
        raw_remap.annotations.rename({k: v for k, v in remap.items() if k in present})

        descriptions = [str(d) for d in raw_remap.annotations.description]
        is_event = np.isin(descriptions, targets)

        if self.tshift != 0:
            shifted = raw_remap.annotations.copy()
            shifted.onset[is_event] += self.tshift
            raw_remap.set_annotations(shifted)

        raw_remap.filter(l_freq=self.bandpass_erp[0], h_freq=self.bandpass_erp[1])

        # Keep the events of interest and the BAD_ segments (needed for
        # reject_by_annotation); drop every other annotation. The masks are
        # recomputed because shifting onsets can re-sort the annotations.
        descriptions = [str(d) for d in raw_remap.annotations.description]
        is_event = np.isin(descriptions, targets)
        is_bad = np.array(
            [d.lower().startswith(("bad", "edge")) for d in descriptions], dtype=bool
        )
        raw_remap.set_annotations(raw_remap.annotations[is_event | is_bad])
        if not is_event.any():
            raise ValueError(
                f"Condition '{cond}': none of the events {targets} were found "
                "in the annotations."
            )

        events, ids = mne.events_from_annotations(
            raw_remap, event_id={t: i + 1 for i, t in enumerate(targets)}
        )
        missing = [t for t in targets if t not in ids]
        if missing:
            LOGGER.warning(
                f"Condition '{cond}': no events found for {missing}; skipped."
            )
        id_of_interest = {t: ids[t] for t in targets if t in ids}

        epochs = mne.Epochs(
            raw_remap,
            event_id=id_of_interest,
            events=events,
            tmin=self.epoch_tlimes[0],
            tmax=self.epoch_tlimes[1],
            baseline=self.baseline,
        )

        fname_out.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(
            fname_out.with_name(fname_out.name + "_epo.fif.gz"), overwrite=overwrite
        )

        return {label: epochs[label] for label in id_of_interest}
