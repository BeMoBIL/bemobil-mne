"""Epoch preparation: event remapping, filtering, and epoching."""

from __future__ import annotations

import re
from pathlib import Path

import mne


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

    print("Discovered event types:")
    for label, id_ in label_to_id.items():
        print(f"  {id_}: {label}")

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
        Stimulus onset delay in seconds.  Annotation onsets are shifted
        forward by this amount so that ``t=0`` aligns with actual stimulus
        presentation.  Set to ``0`` to disable.
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
            Output stem; ``_epo.fif.gz`` is appended automatically.
        cond : str
            Condition key that must exist in ``self.remaps``.
        overwrite : bool
            Overwrite existing epoch file.

        Returns
        -------
        epochs_dict : dict[str, mne.Epochs]
            ``{label: epochs}`` for each target label in the condition remap.
        """
        fname_out = Path(fname_out).with_suffix("")
        remap = self.remaps[cond]

        rename_map, _ = get_stimulus_rename_map(raw_clean.annotations.description)

        raw_remap = raw_clean.copy()
        raw_remap.annotations.rename(rename_map)
        raw_remap.annotations.rename(remap)

        if self.tshift != 0:
            shifted = raw_remap.annotations.copy()
            shifted.onset += self.tshift
            raw_remap.set_annotations(shifted)

        raw_remap.filter(l_freq=self.bandpass_erp[0], h_freq=self.bandpass_erp[1])

        mask = [a in remap.values() for a in raw_remap.annotations.description]
        raw_remap.set_annotations(raw_remap.annotations[mask])

        events, ids = mne.events_from_annotations(raw_remap)
        id_of_interest = {ev: ids[ev] for ev in remap.values()}

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

        return {label: epochs[label] for label in remap.values()}
