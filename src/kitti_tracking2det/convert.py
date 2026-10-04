"""Convert KITTI Tracking layout into per-sequence KITTI Detection layout."""

from __future__ import annotations

import filecmp
import glob
import shutil
from pathlib import Path
from typing import Optional, Sequence

from .constants import TEST_MODALITIES, TRAIN_MODALITIES


def _log(msg: str) -> None:
    print(msg, flush=True)


def _dirs_equal(dir1: Path, dir2: Path) -> bool:
    if not dir1.is_dir() or not dir2.is_dir():
        return False
    comparison = filecmp.dircmp(dir1, dir2)
    if (
        comparison.left_only
        or comparison.right_only
        or comparison.diff_files
        or comparison.funny_files
    ):
        return False
    for common in comparison.common_dirs:
        if not _dirs_equal(dir1 / common, dir2 / common):
            return False
    return True


def _process_kitti_labels(label_file: Path, output_dir: Path) -> None:
    """Split a tracking label file into per-frame detection labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[int, list[str]] = {}
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        frame_number = int(parts[0])
        # Drop frame + track id; keep class + remaining KITTI detection fields.
        category_data = " ".join(parts[2:])
        frames.setdefault(frame_number, []).append(category_data)

    for frame_number, rows in frames.items():
        out = output_dir / f"{frame_number:06d}.txt"
        out.write_text("\n".join(rows) + "\n")


def _copy_file_or_dir(src: Path, dst: Path, *, dry_run: bool = False) -> None:
    if src.is_file():
        if "label" in src.as_posix():
            if dry_run:
                dst.mkdir(parents=True, exist_ok=True)
                return
            _process_kitti_labels(src, dst)
            return
        if "calib" in src.as_posix():
            dst.mkdir(parents=True, exist_ok=True)
            target = dst / src.name
            if not dry_run:
                shutil.copy2(src, target)
            return
        raise ValueError(f"Unexpected file source: {src}")

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        if not dry_run and _dirs_equal(src, dst):
            return
        for item in src.iterdir():
            if item.is_file():
                if dry_run:
                    continue
                shutil.copy2(item, dst / item.name)
        return

    raise FileNotFoundError(f"Source not found: {src}")


def _make_paths(
    tracking_root: Path,
    detection_root: Path,
    phase: str,
    trip_id: str,
    modality: str,
) -> tuple[Path, Path]:
    if modality in ("label_2", "calib"):
        src = tracking_root / phase / modality / f"{trip_id}.txt"
        dst = detection_root / trip_id / phase / modality
        dst.parent.mkdir(parents=True, exist_ok=True)
    else:
        src = tracking_root / phase / modality / trip_id
        dst = detection_root / trip_id / phase / modality
        dst.mkdir(parents=True, exist_ok=True)
    return src, dst


def _create_imagesets_folder(detection_root: Path, trip_id: str) -> Path:
    image_sets = detection_root / trip_id / "ImageSets"
    image_sets.mkdir(parents=True, exist_ok=True)
    for name in ("train.txt", "test.txt", "val.txt", "trainval.txt"):
        (image_sets / name).touch()
    return image_sets


def _replicate_calib(calib_seq_file: Path, image_dir: Path) -> None:
    """Copy sequence calib once per frame, named like the image stems."""
    if not calib_seq_file.is_file():
        return
    frames = sorted(image_dir.glob("*"))
    for frame in frames:
        shutil.copy2(calib_seq_file, calib_seq_file.parent / f"{frame.stem}.txt")
    calib_seq_file.unlink()


def _create_missing_labels(calib_dir: Path, label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    label_names = {p.name for p in label_dir.iterdir() if p.is_file()}
    for calib in calib_dir.iterdir():
        if calib.is_file() and calib.name not in label_names:
            (label_dir / calib.name).touch()


def _write_imagesets(detection_root: Path, phase: str, trip_id: str) -> None:
    image_sets = detection_root / trip_id / "ImageSets"
    pattern = str(detection_root / trip_id / phase / "image_2" / "*")
    image_list = sorted(glob.glob(pattern))
    stems = [Path(p).stem for p in image_list]

    if phase == "training":
        names = ("train.txt", "val.txt", "trainval.txt")
    else:
        names = ("test.txt",)

    for name in names:
        with open(image_sets / name, "w", encoding="utf-8") as fh:
            for stem in stems:
                fh.write(stem + "\n")


def _list_common_sequences(tracking_root: Path) -> list[str]:
    train_dir = tracking_root / "training" / "image_2"
    test_dir = tracking_root / "testing" / "image_2"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {train_dir} and {test_dir}. "
            "Run download first (and ensure image_02 was renamed to image_2)."
        )
    train_ids = {p.name for p in train_dir.iterdir() if p.is_dir()}
    test_ids = {p.name for p in test_dir.iterdir() if p.is_dir()}
    return sorted(train_ids & test_ids)


def convert_tracking_to_detection(
    tracking_root: str | Path,
    detection_root: str | Path,
    *,
    max_sequences: Optional[int] = None,
    sequences: Optional[Sequence[str]] = None,
    include_right_camera: bool = True,
    dry_run: bool = False,
) -> list[str]:
    """
    Convert KITTI Tracking under ``tracking_root`` into per-sequence
    detection folders under ``detection_root``.

    Returns the list of sequence IDs that were processed.
    """
    src_root = Path(tracking_root).expanduser().resolve()
    dst_root = Path(detection_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    trip_ids = _list_common_sequences(src_root)
    if sequences is not None:
        wanted = set(sequences)
        trip_ids = [t for t in trip_ids if t in wanted]
    if max_sequences is not None:
        trip_ids = trip_ids[: max(0, max_sequences)]

    train_mods = list(TRAIN_MODALITIES)
    test_mods = list(TEST_MODALITIES)
    if not include_right_camera:
        train_mods = [m for m in train_mods if m != "image_3"]
        test_mods = [m for m in test_mods if m != "image_3"]

    _log(f"Converting {len(trip_ids)} sequences → {dst_root}")

    for phase, modalities in (("training", train_mods), ("testing", test_mods)):
        for trip_id in trip_ids:
            _log(f"  {phase} / {trip_id}")
            _create_imagesets_folder(dst_root, trip_id)
            for modality in modalities:
                src, dst = _make_paths(src_root, dst_root, phase, trip_id, modality)
                if not src.exists():
                    if modality == "image_3":
                        _log(f"    skip missing {modality}")
                        continue
                    raise FileNotFoundError(f"Missing source: {src}")
                _copy_file_or_dir(src, dst, dry_run=dry_run)

            if dry_run:
                continue

            calib_dir = dst_root / trip_id / phase / "calib"
            seq_calib = calib_dir / f"{trip_id}.txt"
            image_dir = dst_root / trip_id / phase / "image_2"
            _replicate_calib(seq_calib, image_dir)
            _write_imagesets(dst_root, phase, trip_id)

            if phase == "training":
                label_dir = dst_root / trip_id / phase / "label_2"
                _create_missing_labels(calib_dir, label_dir)

    _log("Conversion complete.")
    return trip_ids
