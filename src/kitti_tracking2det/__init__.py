"""Download KITTI Tracking and convert it to KITTI Detection layout."""

from .convert import convert_tracking_to_detection
from .download import download_tracking_dataset

__all__ = [
    "convert_tracking_to_detection",
    "download_tracking_dataset",
    "__version__",
]

__version__ = "0.1.0"
