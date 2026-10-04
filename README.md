# kitti-tracking2det

Download the **KITTI Tracking** dataset and convert it into a **per-sequence KITTI Detection** layout that works with MMDetection3D-style pipelines and other detection codebases.

Zero runtime dependencies (Python 3.8+ stdlib only).

## Requirements

- Python 3.8+
- ~100–130 GB free disk for the tracking download/extract
- Another ~100–130 GB if you keep a full converted detection copy (files are copied, not symlinked)
- `curl` or `wget` recommended for resumable downloads (stdlib `urllib` is used as fallback)

## Install (git clone)

Not published to PyPI — clone and install editable so you can point downloads at your own data directories:

```bash
git clone https://github.com/shahriar0651/kitti-tracking2det.git
cd kitti-tracking2det
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run locally (download + create detection datasets)

You must choose where data lives via `--tracking-root` and `--detection-root` (examples below use `~/datasets/kitti/...`).

One-shot (recommended):

```bash
kitti-tracking2det prepare \
  --tracking-root ~/datasets/kitti/orgTracking \
  --detection-root ~/datasets/kitti/detection
```

Or step by step:

```bash
# 1) Download + extract official KITTI Tracking archives
kitti-tracking2det download \
  --tracking-root ~/datasets/kitti/orgTracking

# 2) Convert tracking layout → per-sequence detection layout
kitti-tracking2det convert \
  --tracking-root ~/datasets/kitti/orgTracking \
  --detection-root ~/datasets/kitti/detection
```

### Useful options

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-sequences N` | all | Limit how many sequences to convert (good for a smoke test) |
| `--sequences 0000,0001` | all common | Convert only these sequence IDs |
| `--no-right-camera` | off | Skip right color camera (`image_3`) — saves space |
| `--dry-run` | off | Build dirs / ImageSets without copying heavy files |
| `--keep-zips` | off | Keep downloaded `.zip` files after extract |

### Smoke test (small / fast)

```bash
# Convert only the first sequence after a full (or partial) download
kitti-tracking2det convert \
  --tracking-root ~/datasets/kitti/orgTracking \
  --detection-root ~/datasets/kitti/detection \
  --max-sequences 1
```

To skip the right camera entirely during download + convert:

```bash
kitti-tracking2det download \
  --tracking-root ~/datasets/kitti/orgTracking \
  --archives data_tracking_image_2.zip,data_tracking_velodyne.zip,data_tracking_label_2.zip,data_tracking_calib.zip

kitti-tracking2det convert \
  --tracking-root ~/datasets/kitti/orgTracking \
  --detection-root ~/datasets/kitti/detection \
  --no-right-camera
```

## Output layout

**Tracking root** (after download):

```
orgTracking/
  training/{image_2,image_3,velodyne,label_2,calib}/<seq>/
  testing/{image_2,image_3,velodyne,calib}/<seq>/
```

**Detection root** (after convert) — one KITTI-style tree per sequence:

```
detection/
  0000/
    ImageSets/{train,val,trainval,test}.txt
    training/{image_2,image_3,velodyne,label_2,calib}/
    testing/{image_2,image_3,velodyne,calib}/
  0001/
    ...
```

- `image_2` = left color camera
- `image_3` = right color camera
- Tracking labels are split into per-frame detection labels
- Calib is replicated once per frame
- Empty label files are created for frames with no objects

## Python API

```python
from kitti_tracking2det import download_tracking_dataset, convert_tracking_to_detection

download_tracking_dataset("~/datasets/kitti/orgTracking")
convert_tracking_to_detection(
    tracking_root="~/datasets/kitti/orgTracking",
    detection_root="~/datasets/kitti/detection",
    max_sequences=2,
)
```

## Notes

- Official archives: [KITTI Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
- Creating MMDetection3D `.pkl` info files is **out of scope** here; run `tools/create_data.py kitti` against each sequence folder after convert.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
