"""Download and extract the official KITTI Tracking archives."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .constants import KITTI_S3_BASE, RENAME_MAP, TRACKING_ARCHIVES


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, resuming when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
        _log(f"  curl → {dest.name}")
        subprocess.run(cmd, check=True)
        return

    if shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(dest), url]
        _log(f"  wget → {dest.name}")
        subprocess.run(cmd, check=True)
        return

    # Stdlib fallback (no resume).
    _log(f"  urllib → {dest.name}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    _log(f"  unzip {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _rename_kitti_dirs(tracking_root: Path) -> None:
    for src_rel, dst_rel in RENAME_MAP:
        src = tracking_root / src_rel
        dst = tracking_root / dst_rel
        if src.is_dir() and not dst.exists():
            src.rename(dst)
            _log(f"  renamed {src_rel} → {dst_rel}")
        elif dst.is_dir():
            _log(f"  skip rename (exists): {dst_rel}")


def download_tracking_dataset(
    tracking_root: str | Path,
    *,
    archives: Optional[Sequence[str]] = None,
    keep_zips: bool = False,
    skip_extract: bool = False,
) -> Path:
    """
    Download KITTI Tracking archives into ``tracking_root``, extract them,
    and rename ``image_02`` / ``label_02`` → ``image_2`` / ``label_2``
    (and the right-camera ``image_03`` → ``image_3``).

    Returns the resolved tracking root path.
    """
    root = Path(tracking_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    selected = list(archives) if archives is not None else list(TRACKING_ARCHIVES)
    _log(f"Downloading KITTI Tracking → {root}")

    for name in selected:
        url = f"{KITTI_S3_BASE}/{name}"
        zip_path = root / name
        if zip_path.exists() and zip_path.stat().st_size > 0:
            _log(f"  already present: {name}")
        else:
            _download_file(url, zip_path)

        if not skip_extract:
            _extract_zip(zip_path, root)
            if not keep_zips:
                zip_path.unlink(missing_ok=True)
                _log(f"  removed {name}")

    _rename_kitti_dirs(root)
    _log("Download complete.")
    return root


def list_default_archives() -> Iterable[str]:
    return TRACKING_ARCHIVES


def _cli_download(argv: Optional[Sequence[str]] = None) -> int:
    """Internal helper used by tests; prefer ``kitti_tracking2det.cli``."""
    from .cli import main

    return main(["download", *(argv or [])])


if __name__ == "__main__":
    sys.exit(_cli_download(sys.argv[1:]))
