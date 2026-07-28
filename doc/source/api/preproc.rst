.. _api-preproc:

Preprocessing
=============

The preprocessing module covers the full EEG pipeline, epoch preparation,
rigid-body motion kinematics, and a set of lower-level signal-processing
utilities that the pipeline calls internally but which are also usable
standalone.

----

EEG pipeline
------------

:class:`~bemobil_mne.preproc.EEGPreprocessor` orchestrates the full pipeline:
ZapLine → bad-channel detection → bandpass filter → ASR → ICA/AMICA →
ICLabel classification → dipole fitting → average re-reference → bad-channel
interpolation.  All steps are recorded in a provenance descriptor stored on
the :class:`mne.io.Raw` object.

.. currentmodule:: bemobil_mne.preproc

.. autosummary::
   :toctree: generated/

   EEGPreprocessor

----

Bad channel detection
---------------------

Standalone bad-channel detector that combines PyPREP, FASTER, per-channel
flatline detection, and a line-noise z-score criterion.  Called automatically
by :class:`EEGPreprocessor` but can also be run on any
:class:`mne.io.Raw` object directly.

.. autosummary::
   :toctree: generated/

   get_bad_chs

----

Signal cleaning
---------------

Individual signal-processing steps that :class:`EEGPreprocessor` calls
internally.  Useful when building a custom pipeline or applying a single step
in isolation.

.. autosummary::
   :toctree: generated/

   compute_zapline
   compute_asr
   compute_ica
   detect_bad_by_line_noise
   compute_mi_reduction
   get_raw_subset

----

Dipole fitting
--------------

.. autosummary::
   :toctree: generated/

   fit_dipoles_on_ica
   compute_dipolarity
   auto_coreg_fsaverage

----

Epoching
--------

Helpers for converting continuous preprocessed data into epochs, including
stimulus-label renaming utilities for common BPN paradigms.

.. autosummary::
   :toctree: generated/

   EpochPreparer
   get_stimulus_rename_map

----

Motion capture
--------------

Rigid-body kinematics from XDF motion streams.  Works on the motion data
returned by :class:`~bemobil_mne.io.XDFLoader` in
:attr:`~bemobil_mne.io.MultimodalRecording.motion`.

.. autosummary::
   :toctree: generated/

   find_rigid_bodies
   process_rigid_body
   split_by_rigid_body

----

Provenance
----------

Lightweight provenance tracking: each pipeline step appends a JSON-serialisable
descriptor to :attr:`mne.Info.description` so that any saved file carries a
record of exactly how it was processed.

.. autosummary::
   :toctree: generated/

   init_descriptor
   set_descriptor
   get_descriptor
   append_desc
   sig_params

----

Utilities
---------

.. autosummary::
   :toctree: generated/

   StepTimer
   format_duration
   build_sys_info
