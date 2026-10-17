PyBeMoBIL
============

**PyBeMoBIL** is a Python library for EEG and multimodal Mobile Brain/Body Imaging (MoBI)
data processing, developed at the `Biopsychology & Neuroergonomics department
<https://www.bpn.tu-berlin.de>`_ at TU Berlin.

It provides a full analysis stack built on top of `MNE-Python <https://mne.tools>`_:

- **Multimodal loading**  -  read XDF recordings and align auxiliary streams (ECG, gaze, EMG, EDA)
  to a common time grid using the :class:`~pybemobil.io.XDFLoader`.
- **EEG preprocessing pipeline**  -  ZapLine line-noise removal → bad channel detection →
  bandpass filter → optional ASR → JAMICA/ICA → ICLabel classification → dipole fitting, all in
  one :class:`~pybemobil.preproc.EEGPreprocessor` call with full provenance tracking.
- **BIDS export**  -  write cleaned datasets to the BIDS standard via
  :func:`~pybemobil.io.export_to_bids`.
- **Motion capture**  -  rigid-body kinematics from XDF motion streams via
  :func:`~pybemobil.preproc.find_rigid_bodies`.
- **Visualization**  -  ERP, PSD, TFR, and topomap plots built on MNE and matplotlib.

The pipeline is designed to replicate and extend the MATLAB-based
`BeMoBIL pipeline <https://github.com/BeMoBIL/bemobil-pipeline>`_ in Python,
sharing the same preprocessing defaults where meaningful.
See :doc:`guides/bemobil` for a full comparison and migration example.

----

Installation
------------

Install the latest release from PyPI:

.. code-block:: bash

   pip install pybemobil

To also install the JAX-accelerated AMICA backend (the default ICA method,
which falls back to Picard when this is missing):

.. code-block:: bash

   pip install "pybemobil[jamica]"

Alternatively, install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/BeMoBIL/pybemobil.git

or install from source in editable mode, for development:

.. code-block:: bash

   git clone https://github.com/BeMoBIL/pybemobil.git
   cd pybemobil
   pip install -e ".[dev]"

We recommend installing into a fresh environment, e.g.
``conda create -n pybemobil python=3.12``.

----

Quick start
-----------

.. code-block:: python

   from pybemobil.io import XDFLoader
   from pybemobil.preproc import EEGPreprocessor

   # 1. Load a multimodal XDF recording
   loader = XDFLoader(
       montage="easycap-M1",
       target_sfreq=250.0,
   )
   recording = loader.load("sub-01_task-walk_run-01.xdf")
   raw = recording.raw          # mne.io.Raw with EEG + aux channels merged

   # 2. Run the full preprocessing pipeline
   preprocessor = EEGPreprocessor(
       loader=loader,
       line_noise_freq="europe",   # 50 Hz + harmonics up to Nyquist
       line_noise_method="adaptive",  # ZapLine-Plus before filtering
   )
   raw_clean, report, metadata = preprocessor.run_raw(
       raw,
       fname_out="derivatives/pybemobil/sub-01/eeg/sub-01_task-walk_run-01_eeg.fif",
   )
   # writes derivatives/.../sub-01_task-walk_run-01_desc-clean_eeg.fif (+ ICA,
   # report, ...) and the untouched input to sub-01/eeg/ (BIDS raw)

   # raw_clean : ICA-cleaned, average-referenced, bad channels interpolated
   # report    : mne.Report HTML summary (saved alongside output)
   # metadata  : dict with raw_source, raw_pre_ica, ica, ic_labels, bad_ch_dict, …

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
