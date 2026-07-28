.. _guide-bemobil:

Comparison with the BeMoBIL pipeline
=====================================

The `BeMoBIL pipeline <https://github.com/BeMoBIL/bemobil-pipeline>`_ (Klug et al., 2022)
is a widely-used MATLAB toolbox for automated MoBI data analysis.  BeMoBIL-Python is its
Python/MNE counterpart, designed to produce equivalent results while integrating with the
broader scientific Python ecosystem.

This page describes what the two pipelines share, where they differ and why, and shows a
side-by-side minimal example.

----

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - MATLAB BeMoBIL pipeline
     - BeMoBIL-Python
   * - Language
     - MATLAB
     - Python
   * - EEG framework
     - EEGLAB
     - MNE-Python
   * - ICA
     - AMICA (native MATLAB)
     - AMICA via ``amica-python`` (falls back to picard)
   * - IC classification
     - ICLabel (EEGLAB plugin)
     - ``mne-icalabel``
   * - Line-noise removal
     - ZapLine-Plus (auto-detect)
     - ZapLine-Plus via ``mne-denoise`` (``zapline_method="adaptive"``)
   * - Bad channel detection
     - ``clean_rawdata`` (ASR-based correlation)
     - PyPREP + FASTER + optional line-noise / flatline criteria
   * - ASR
     - Not used (AMICA autoreject instead)
     - Optional (``asr=False`` by default)
   * - BIDS support
     - Manual / separate toolbox
     - Built-in via ``mne-bids``
   * - Provenance
     - ``EEG.etc`` struct fields
     - JSON descriptor in ``raw.info["description"]``
   * - Output format
     - ``.set`` / ``.fdt`` (EEGLAB)
     - ``.fif.gz`` (MNE)

----

Shared defaults
---------------

The following preprocessing parameters are set to match the original MATLAB implementation's published defaults
(``bemobil_check_config.m``):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Setting
     - MATLAB BeMoBIL
     - BeMoBIL-Python
   * - Resample for ICA
     - 250 Hz
     - ``downsample_ica=250.0``
   * - ICA high-pass filter
     - 1.75 Hz
     - ``filter_bands_ica=(1.75, None)``
   * - ZapLine mode
     - auto-detect (``noisefreqs=[]``)
     - ``zapline_method="adaptive"``
   * - ICLabel decision rule
     - popularity vote (``iclabel_threshold=-1``)
     - ``thresh=-1``
   * - ICs kept
     - all except Eye (classes 1,2,4,5,6,7)
     - ``include_labels`` = all except ``"eye blink"``
   * - Line-noise bad-ch criterion
     - off
     - ``line_noise_crit=None``
   * - Flatline bad-ch criterion
     - off
     - ``flatline_crit=None``
   * - Dipole RV threshold
     - 100 % (no threshold)
     - ``rv_thresh=None``
   * - Remove outside-head dipoles
     - off
     - ``remove_outside_head=False``

----

Intentional differences
------------------------

**Bad channel detection method**
   The original MATLAB implementation uses ``clean_rawdata`` (a variant of the ASR correlation algorithm) with
   ``chancorr_crit=0.8`` and ``chan_max_broken_time=0.3``.  BeMoBIL-Python uses a combination
   of PyPREP and FASTER, which are well-validated standalone Python implementations.
   Both approaches flag channels that decorrelate from their neighbours; the underlying
   principle is the same.

**ASR**
   The original MATLAB implementation does not apply ASR - it relies on AMICA's built-in autoreject
   (``AMICA_autoreject=1``, sigma threshold 3).  BeMoBIL-Python also disables ASR by default
   (``asr=False``) for the same reason: AMICA's internal rejection is sufficient for clean
   ICA decompositions.  ASR is available as an opt-in (``asr=True`` or ``asr={"cutoff": 20}``)
   for pipelines that do not use AMICA.

**Final high-pass filter**
   The original MATLAB implementation applies a 0.2 Hz high-pass after ICA (``final_filter_lower_edge=0.2``).
   BeMoBIL-Python removed this step (``final_filter_bands`` was dropped) because the
   main bandpass filter (``filter_bands=(0.1, 100.0)`` by default) already removes
   sub-0.1 Hz drift, and the final filter adds an implicit dependency on analysis choices
   that belong downstream.  Add it back explicitly if needed:

   .. code-block:: python

      raw_clean.filter(l_freq=0.2, h_freq=None)

**Output format**
   The original MATLAB implementation saves ``.set`` / ``.fdt`` (EEGLAB) files.  BeMoBIL-Python saves ``.fif.gz``
   (MNE native), which supports lossless compression and carries full metadata including
   the provenance descriptor.

----

Minimal example
---------------

The following script replicates the core BeMoBIL EEG preprocessing steps using BeMoBIL-Python.

.. code-block:: python

   from bemobil_python.io import XDFLoader
   from bemobil_python.preproc import EEGPreprocessor

   # --- Equivalent to bemobil_process_all_EEG_preprocessing ---

   loader = XDFLoader(
       montage="standard_1020",       # channel_locations_filename
       target_sfreq=250.0,            # resample_freq = 250
       old_reference="FCz",           # ref_channel = 'FCz'
   )

   preprocessor = EEGPreprocessor(
       loader=loader,
       line_noise_freq="europe",      # 50 Hz; use "usa" for 60 Hz
       zapline_method="adaptive",     # zaplineConfig.noisefreqs = []
       filter_bands=(0.1, 100.0),     # broad bandpass (BeMoBIL keeps full band)
       filter_bands_ica=(1.75, None), # filter_lowCutoffFreqAMICA = 1.75
       downsample_ica=250.0,          # resample_freq = 250
       ica_method="amica",            # AMICA
       thresh=-1,                     # iclabel_threshold = -1
       # include_labels default = all except eye blink (iclabel_classes = [1 2 4 5 6 7])
       fit_dipoles=True,              # dipole fitting on
       rv_thresh=None,                # residualVariance_threshold = 100 (no threshold)
       remove_outside_head=False,     # do_remove_outside_head = 'off'
   )

   raw_clean, report, metadata = preprocessor.run(
       "sub-01_task-walk_run-01.xdf",
       fname_out="sub-01_preprocessed_and_ICA.fif.gz",
       overwrite=True,
   )

   # Access pipeline outputs
   ica        = metadata["ica"]
   ic_labels  = metadata["ic_labels"]
   dipoles    = metadata["dipoles"]
   bad_ch_dict = metadata["bad_ch_dict"]

----

Reference
---------

Klug, M., Jeung, S., Wunderlich, A., Gehrke, L., Protzak, J., Djebbara, Z.,
Argubi-Wollesen, A., Wollesen, B., & Gramann, K. (2022).
*The BeMoBIL Pipeline for automated analyses of multimodal mobile brain and body
imaging data.* bioRxiv. https://doi.org/10.1101/2022.09.29.510051
