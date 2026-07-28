BeMoBIL-MNE
============

**BeMoBIL-MNE** is a Python library for EEG and multimodal Mobile Brain/Body Imaging (MoBI)
data processing, developed at the `Biopsychology & Neuroergonomics department
<https://www.bpn.tu-berlin.de>`_ at TU Berlin.

It provides a full analysis stack built on top of `MNE-Python <https://mne.tools>`_:

- **Multimodal loading**  -  read XDF recordings and align auxiliary streams (ECG, gaze, EMG, EDA)
  to a common time grid using the :class:`~bemobil_mne.io.XDFLoader`.
- **EEG preprocessing pipeline**  -  ZapLine line-noise removal → bad channel detection →
  bandpass filter → optional ASR → AMICA/ICA → ICLabel classification → dipole fitting, all in
  one :class:`~bemobil_mne.preproc.EEGPreprocessor` call with full provenance tracking.
- **BIDS export**  -  write cleaned datasets to the BIDS standard via
  :func:`~bemobil_mne.io.export_to_bids`.
- **Motion capture**  -  rigid-body kinematics from XDF motion streams via
  :func:`~bemobil_mne.preproc.find_rigid_bodies`.
- **Visualization**  -  ERP, PSD, TFR, and topomap plots built on MNE and matplotlib.

The pipeline is designed to replicate and extend the MATLAB-based
`BeMoBIL pipeline <https://github.com/BeMoBIL/bemobil-pipeline>`_ in Python,
sharing the same preprocessing defaults where meaningful.
See :doc:`guides/bemobil` for a full comparison and migration example.

----

Installation
------------

.. code-block:: bash

   git clone https://github.com/BeMoBIL/bemobil-mne.git
   cd bemobil-mne
   pip install -e .

----

Quick start
-----------

.. code-block:: python

   from bemobil_mne.io import XDFLoader
   from bemobil_mne.preproc import EEGPreprocessor

   # 1. Load a multimodal XDF recording
   loader = XDFLoader(
       montage="standard_1020",
       target_sfreq=250.0,
   )
   recording = loader.load("sub-01_task-walk_run-01.xdf")
   raw = recording.raw          # mne.io.Raw with EEG + aux channels merged

   # 2. Run the full preprocessing pipeline
   preprocessor = EEGPreprocessor(
       loader=loader,
       line_noise_freq="europe",   # 50 Hz + harmonics up to Nyquist
       zapline_method="adaptive",  # ZapLine-Plus before filtering
   )
   raw_clean, report, metadata = preprocessor.run_raw(raw, fname_out="sub-01_clean.fif.gz")

   # raw_clean : ICA-cleaned, average-referenced, bad channels interpolated
   # report    : mne.Report HTML summary (saved alongside output)
   # metadata  : dict with raw_minimal, raw_asr, ica, ic_labels, bad_ch_dict, …

----

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/io
   api/preproc
   api/viz

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/bemobil
