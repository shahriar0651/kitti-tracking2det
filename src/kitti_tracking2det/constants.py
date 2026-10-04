"""KITTI Tracking download URLs and naming conventions."""

from __future__ import annotations

KITTI_S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

# Official KITTI Tracking archives.
# image_2 = left color camera, image_3 = right color camera.
TRACKING_ARCHIVES: tuple[str, ...] = (
    "data_tracking_image_2.zip",
    "data_tracking_image_3.zip",
    "data_tracking_velodyne.zip",
    "data_tracking_label_2.zip",
    "data_tracking_calib.zip",
)

# After unzip, KITTI uses image_02 / label_02; detection layouts use image_2 / label_2.
RENAME_MAP: tuple[tuple[str, str], ...] = (
    ("training/image_02", "training/image_2"),
    ("training/image_03", "training/image_3"),
    ("training/label_02", "training/label_2"),
    ("testing/image_02", "testing/image_2"),
    ("testing/image_03", "testing/image_3"),
)

TRAIN_MODALITIES: tuple[str, ...] = (
    "calib",
    "image_2",
    "image_3",
    "label_2",
    "velodyne",
)
TEST_MODALITIES: tuple[str, ...] = (
    "calib",
    "image_2",
    "image_3",
    "velodyne",
)
