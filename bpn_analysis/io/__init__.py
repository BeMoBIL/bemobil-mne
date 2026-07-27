"""I/O utilities for BPN-analysis."""

from bpn_analysis.io.alignment import align_stream_to_timestamps
from bpn_analysis.io.bids_export import (
    batch_export_to_bids,
    export_to_bids,
    make_bids_dataset_description,
)
from bpn_analysis.io.xdf import MultimodalRecording, XDFLoader

__all__ = [
    "XDFLoader",
    "MultimodalRecording",
    "align_stream_to_timestamps",
    # BIDS export
    "export_to_bids",
    "make_bids_dataset_description",
    "batch_export_to_bids",
]
