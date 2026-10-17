"""Preprocessing utilities for PyBeMoBIL."""

from pybemobil.preproc.epoching import EpochPreparer, get_stimulus_rename_map
from pybemobil.preproc.motion import (
    find_rigid_bodies,
    process_rigid_body,
    split_by_rigid_body,
)
from pybemobil.preproc.make_report import make_report
from pybemobil.preproc.preprocessing import (
    EEGPreprocessor,
    NumpyEncoder,
    get_bad_chs,
    get_derivative_paths,
)
from pybemobil.preproc.utils import (
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
    remove_line,
    detect_bad_by_line_noise,
    fit_dipoles_on_ica,
    format_duration,
    get_descriptor,
    init_descriptor,
    set_descriptor,
    sig_params,
)

__all__ = [
    # Classes
    "EEGPreprocessor",
    "get_derivative_paths",
    # Report
    "make_report",
    "EpochPreparer",
    "NumpyEncoder",
    "StepTimer",
    # Preprocessing functions
    "get_bad_chs",
    "compute_asr",
    "remove_line",
    "compute_ica",
    "fit_dipoles_on_ica",
    "compute_dipolarity",
    "compute_mi_reduction",
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
    # Epoching
    "get_stimulus_rename_map",
]
