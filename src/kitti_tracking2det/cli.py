"""Command-line interface for kitti-tracking2det."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import __version__
from .constants import TRACKING_ARCHIVES
from .convert import convert_tracking_to_detection
from .download import download_tracking_dataset


def _parse_sequences(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kitti-tracking2det",
        description=(
            "Download KITTI Tracking and convert it to per-sequence "
            "KITTI Detection layout."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download + extract KITTI Tracking")
    p_dl.add_argument(
        "--tracking-root",
        required=True,
        help="Directory for the original tracking dataset",
    )
    p_dl.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip archives after extraction",
    )
    p_dl.add_argument(
        "--archives",
        default=None,
        help=(
            "Comma-separated archive names to download "
            f"(default: all of {', '.join(TRACKING_ARCHIVES)})"
        ),
    )

    # convert
    p_cv = sub.add_parser(
        "convert",
        help="Convert tracking layout → per-sequence detection layout",
    )
    p_cv.add_argument("--tracking-root", required=True)
    p_cv.add_argument("--detection-root", required=True)
    p_cv.add_argument("--max-sequences", type=int, default=None)
    p_cv.add_argument(
        "--sequences",
        default=None,
        help="Comma-separated sequence IDs (e.g. 0000,0001)",
    )
    p_cv.add_argument(
        "--no-right-camera",
        action="store_true",
        help="Do not copy image_3 (right color camera)",
    )
    p_cv.add_argument(
        "--dry-run",
        action="store_true",
        help="Create folders / ImageSets without copying heavy files",
    )

    # prepare = download + convert
    p_pr = sub.add_parser("prepare", help="Download then convert")
    p_pr.add_argument("--tracking-root", required=True)
    p_pr.add_argument("--detection-root", required=True)
    p_pr.add_argument("--keep-zips", action="store_true")
    p_pr.add_argument("--max-sequences", type=int, default=None)
    p_pr.add_argument("--sequences", default=None)
    p_pr.add_argument("--no-right-camera", action="store_true")
    p_pr.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "download":
        archives = None
        if args.archives:
            archives = [a.strip() for a in args.archives.split(",") if a.strip()]
        download_tracking_dataset(
            args.tracking_root,
            archives=archives,
            keep_zips=args.keep_zips,
        )
        return 0

    if args.command == "convert":
        convert_tracking_to_detection(
            args.tracking_root,
            args.detection_root,
            max_sequences=args.max_sequences,
            sequences=_parse_sequences(args.sequences),
            include_right_camera=not args.no_right_camera,
            dry_run=args.dry_run,
        )
        return 0

    if args.command == "prepare":
        download_tracking_dataset(
            args.tracking_root,
            keep_zips=args.keep_zips,
        )
        convert_tracking_to_detection(
            args.tracking_root,
            args.detection_root,
            max_sequences=args.max_sequences,
            sequences=_parse_sequences(args.sequences),
            include_right_camera=not args.no_right_camera,
            dry_run=args.dry_run,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
