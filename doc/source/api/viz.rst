.. _api-viz:

Visualization
=============

Plotting utilities built on MNE-Python and matplotlib.  All functions accept
dictionaries of :class:`mne.Evoked` or :class:`mne.Epochs` objects keyed by
condition name, making it straightforward to overlay or compare multiple
conditions in a single call.

----

Power spectral density
-----------------------

.. currentmodule:: bemobil_python.viz

.. autosummary::
   :toctree: generated/

   plot_PSD
   plot_psd_topomaps

----

Time-frequency
--------------

.. autosummary::
   :toctree: generated/

   plot_TFR
   plot_tfr_topomaps
   plot_ERSP

----

Event-related potentials
------------------------

.. autosummary::
   :toctree: generated/

   plot_ERP
   plot_various_ERPs
   plot_evoked_data
   plot_joint
   plot_topo
   plot_butterfly

----

Utilities
---------

.. autosummary::
   :toctree: generated/

   glue_imgs
