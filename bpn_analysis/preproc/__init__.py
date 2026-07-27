"""Preprocessing utilities for BPN-analysis."""

from bpn_analysis.preproc.epoching import EpochPreparer, get_stimulus_rename_map
from bpn_analysis.preproc.motion import (
    find_rigid_bodies,
    process_rigid_body,
    split_by_rigid_body,
)
from bpn_analysis.preproc.preprocessing import (
    EEGPreprocessor,
    NumpyEncoder,
    get_bad_chs,
)
from bpn_analysis.preproc.utils import (
    SUBSET_CHANNELS,
    StepTimer,
    _annotate_break_iter,
    _handle_trans,
    append_desc,
    auto_coreg_fsaverage,
    build_sys_info,
    compute_asr,
    compute_dipolarity,
    compute_ica,
    compute_mi_reduction,
    compute_zapline,
    detect_bad_by_line_noise,
    fit_dipoles_on_ica,
    format_duration,
    get_descriptor,
    get_raw_subset,
    init_descriptor,
    set_descriptor,
    sig_params,
)

__all__ = [
    # Classes
    "EEGPreprocessor",
    "EpochPreparer",
    "NumpyEncoder",
    "StepTimer",
    # Preprocessing functions
    "get_bad_chs",
    "compute_asr",
    "compute_zapline",
    "compute_ica",
    "fit_dipoles_on_ica",
    "compute_dipolarity",
    "compute_mi_reduction",
    "get_raw_subset",
    "auto_coreg_fsaverage",
    # Bad channel detection helpers
    "detect_bad_by_line_noise",
    # Motion processing
    "find_rigid_bodies",
    "process_rigid_body",
    "split_by_rigid_body",
    # Provenance
    "init_descriptor",
    "get_descriptor",
    "set_descriptor",
    "append_desc",
    "sig_params",
    # Utilities
    "format_duration",
    "build_sys_info",
    "_annotate_break_iter",
    "_handle_trans",
    "SUBSET_CHANNELS",
    # Epoching
    "get_stimulus_rename_map",
]
