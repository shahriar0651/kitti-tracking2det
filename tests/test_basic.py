"""Lightweight unit tests (no network / no full dataset required)."""

from __future__ import annotations

from pathlib import Path

from kitti_tracking2det.cli import build_parser
from kitti_tracking2det.convert import _process_kitti_labels, convert_tracking_to_detection
from kitti_tracking2det.constants import TRACKING_ARCHIVES


def test_archives_include_both_cameras():
    assert "data_tracking_image_2.zip" in TRACKING_ARCHIVES
    assert "data_tracking_image_3.zip" in TRACKING_ARCHIVES


def test_process_kitti_labels(tmp_path: Path):
    label_file = tmp_path / "0000.txt"
    label_file.write_text(
        "0 1 Car 0 0 0 10 20 30 40 1.5 1.6 3.5 1 2 3 0.1\n"
        "0 2 Pedestrian 0 0 0 11 21 31 41 1.7 0.6 0.8 4 5 6 0.2\n"
        "1 1 Car 0 0 0 12 22 32 42 1.5 1.6 3.5 7 8 9 0.3\n"
    )
    out = tmp_path / "labels"
    _process_kitti_labels(label_file, out)
    assert (out / "000000.txt").exists()
    assert (out / "000001.txt").exists()
    lines0 = (out / "000000.txt").read_text().strip().splitlines()
    assert len(lines0) == 2
    assert lines0[0].startswith("Car ")


def test_convert_dry_run_structure(tmp_path: Path):
    tracking = tmp_path / "org"
    detection = tmp_path / "det"

    for phase in ("training", "testing"):
        for mod in ("image_2", "image_3", "velodyne"):
            seq = tracking / phase / mod / "0000"
            seq.mkdir(parents=True)
            (seq / "000000.png" if mod.startswith("image") else seq / "000000.bin").write_bytes(
                b"x"
            )
        (tracking / phase / "calib").mkdir(parents=True, exist_ok=True)
        (tracking / phase / "calib" / "0000.txt").write_text("P0: 1\n")
    (tracking / "training" / "label_2").mkdir(parents=True)
    (tracking / "training" / "label_2" / "0000.txt").write_text(
        "0 1 Car 0 0 0 10 20 30 40 1.5 1.6 3.5 1 2 3 0.1\n"
    )

    ids = convert_tracking_to_detection(
        tracking,
        detection,
        sequences=["0000"],
        dry_run=False,
    )
    assert ids == ["0000"]
    assert (detection / "0000" / "training" / "image_2" / "000000.png").exists()
    assert (detection / "0000" / "training" / "image_3" / "000000.png").exists()
    assert (detection / "0000" / "training" / "label_2" / "000000.txt").exists()
    assert (detection / "0000" / "ImageSets" / "train.txt").read_text().strip() == "000000"


def test_cli_help():
    parser = build_parser()
    args = parser.parse_args(
        ["convert", "--tracking-root", "/tmp/a", "--detection-root", "/tmp/b"]
    )
    assert args.command == "convert"
