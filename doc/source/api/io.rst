.. _api-io:

I/O
===

Functions and classes for loading multimodal XDF recordings, aligning auxiliary
streams to a common time grid, and exporting datasets to BIDS.

----

Loading
-------

The central entry point for reading XDF files.
:class:`~bemobil_mne.io.XDFLoader` handles stream detection, resampling, and
multi-stream alignment in one call, returning a
:class:`~bemobil_mne.io.MultimodalRecording` that bundles the merged
:class:`mne.io.Raw` object together with any Tier-2 (high-rate) streams and
irregular event streams.

.. currentmodule:: bemobil_mne.io

.. autosummary::
   :toctree: generated/

   XDFLoader
   MultimodalRecording

----

Stream alignment
----------------

Low-level utility for aligning a single auxiliary stream to an existing time
grid.  Used internally by :class:`XDFLoader` but also useful standalone when
combining data from different acquisition systems.

.. autosummary::
   :toctree: generated/

   align_stream_to_timestamps

----

BIDS export
-----------

Wrappers around :mod:`mne_bids` for writing cleaned EEG datasets to the
`Brain Imaging Data Structure <https://bids-specification.readthedocs.io>`_ standard.

.. autosummary::
   :toctree: generated/

   export_to_bids
   batch_export_to_bids
   make_bids_dataset_description
