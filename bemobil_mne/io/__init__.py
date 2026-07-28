"""I/O utilities for BeMoBIL-MNE."""

from bemobil_mne.io.alignment import align_stream_to_timestamps
from bemobil_mne.io.bids_export import (
    batch_export_to_bids,
    export_to_bids,
    make_bids_dataset_description,
)
from bemobil_mne.io.xdf import MultimodalRecording, XDFLoader

__all__ = [
    "XDFLoader",
    "MultimodalRecording",
    "align_stream_to_timestamps",
    # BIDS export
    "export_to_bids",
    "make_bids_dataset_description",
    "batch_export_to_bids",
]
