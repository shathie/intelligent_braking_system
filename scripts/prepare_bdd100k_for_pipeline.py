"""
Convert BDD100K driving videos into the project's custom synchronized format.

Input options:
  1) Single video file
  2) Directory containing multiple video files

Optional labels JSON:
  BDD100K video labels (for weather/timeofday attributes) can be provided to
  improve surface_type tagging.

Output format (compatible with MultiModalBrakingDataset and preprocess_data.py --only-custom):
  <output_dir>/images/<timestamp>_<video_stem>_<frame_idx>.jpg
  <output_dir>/can_data.csv
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

SUPPORTED_VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}


def _collect_videos(video_file: str, video_dir: str) -> List[Path]:
    videos: List[Path] = []

    if video_file:
        p = Path(video_file)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {p}")
        videos.append(p)

    if video_dir:
        d = Path(video_dir)
        if not d.exists():
            raise FileNotFoundError(f"Video directory not found: {d}")
        dir_videos = [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
        videos.extend(dir_videos)

    # De-duplicate while preserving order
    seen = set()
    unique_videos = []
    for v in videos:
        key = str(v.resolve())
        if key not in seen:
            unique_videos.append(v)
            seen.add(key)

    if not unique_videos:
        raise ValueError("No videos found. Provide --video-file or --video-dir with supported extensions.")

    return unique_videos


def _load_bdd_labels_map(labels_json: str) -> Dict[str, Dict]:
    if not labels_json:
        return {}

    path = Path(labels_json)
    if not path.exists():
        raise FileNotFoundError(f"Labels JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: Dict[str, Dict] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            attrs = item.get("attributes", {})
            if name:
                stem = Path(name).stem
                label_map[stem] = attrs if isinstance(attrs, dict) else {}

    return label_map


def _infer_surface_type(default_surface_type: str, attrs: Dict) -> str:
    if not attrs:
        return default_surface_type

    weather = str(attrs.get("weather", "")).strip().lower()
    scene = str(attrs.get("scene", "")).strip().lower()

    if weather in {"rainy", "snowy", "foggy"}:
        return f"bdd_{weather}"
    if scene in {"tunnel", "residential", "city street", "highway"}:
        scene_tag = scene.replace(" ", "_")
        return f"bdd_{scene_tag}"
    return default_surface_type


def _decode_video_to_rows(
    video_path: Path,
    images_dir: Path,
    label_attrs: Dict,
    default_mu: float,
    default_surface_type: str,
    sample_fps: float,
    max_frames_per_video: int,
    resize_width: int,
    resize_height: int,
    start_timestamp: float,
) -> Tuple[List[Dict], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0

    # Sampling strategy: keep roughly sample_fps frames per second
    if sample_fps <= 0:
        frame_stride = 1
    else:
        frame_stride = max(1, int(round(native_fps / sample_fps)))

    frame_index = 0
    kept = 0
    rows: List[Dict] = []

    surface_type = _infer_surface_type(default_surface_type, label_attrs)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        if max_frames_per_video > 0 and kept >= max_frames_per_video:
            break

        if resize_width > 0 and resize_height > 0:
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        timestamp = start_timestamp + (frame_index / native_fps)
        image_name = f"{timestamp:.6f}_{video_path.stem}_{frame_index:06d}.jpg"
        image_path = images_dir / image_name
        cv2.imwrite(str(image_path), frame)

        # BDD videos do not contain CAN; create a scaffold CSV with defaults.
        row = {
            "timestamp": float(timestamp),
            "a_x": 0.0,
            "a_y": 0.0,
            "omega_FL": 0.0,
            "omega_FR": 0.0,
            "omega_RL": 0.0,
            "omega_RR": 0.0,
            "steering_angle": 0.0,
            "brake_pressure": 0.0,
            "v_x": 0.0,
            "yaw_rate": 0.0,
            "tire_temp_FL": 0.0,
            "tire_temp_FR": 0.0,
            "mu": float(default_mu),
            "surface_type": surface_type,
        }
        rows.append(row)

        kept += 1
        frame_index += 1

    cap.release()

    # Continue timeline after this video to avoid duplicate timestamps
    # across multi-video ingestion.
    end_timestamp = start_timestamp + max(kept, 1) * (1.0 / max(sample_fps, 1.0))
    return rows, end_timestamp, kept


def convert_bdd_to_custom(
    video_file: str,
    video_dir: str,
    output_dir: str,
    labels_json: str,
    default_mu: float,
    default_surface_type: str,
    sample_fps: float,
    max_frames_per_video: int,
    resize_width: int,
    resize_height: int,
    clear_output: bool,
) -> None:
    videos = _collect_videos(video_file=video_file, video_dir=video_dir)
    label_map = _load_bdd_labels_map(labels_json)

    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    can_csv_path = out_dir / "can_data.csv"

    if clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    images_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    total_kept = 0
    current_ts = 0.0

    for vid in videos:
        attrs = label_map.get(vid.stem, {})
        rows, current_ts, kept = _decode_video_to_rows(
            video_path=vid,
            images_dir=images_dir,
            label_attrs=attrs,
            default_mu=default_mu,
            default_surface_type=default_surface_type,
            sample_fps=sample_fps,
            max_frames_per_video=max_frames_per_video,
            resize_width=resize_width,
            resize_height=resize_height,
            start_timestamp=current_ts,
        )
        all_rows.extend(rows)
        total_kept += kept
        print(f"[OK] Processed {vid.name}: kept_frames={kept}")

    if not all_rows:
        raise RuntimeError("No frames were extracted from videos.")

    pd.DataFrame(all_rows).to_csv(can_csv_path, index=False)
    print(f"[OK] Wrote images to: {images_dir}")
    print(f"[OK] Wrote CAN scaffold CSV: {can_csv_path}")
    print(f"[OK] Total extracted frames: {total_kept}")
    print("[NEXT] Run: python scripts/preprocess_data.py --only-custom --custom-dir <output_dir>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BDD100K videos for project custom preprocessing.")
    parser.add_argument("--video-file", type=str, default="", help="Path to a single video file.")
    parser.add_argument("--video-dir", type=str, default="", help="Path to a directory containing videos.")
    parser.add_argument("--output-dir", type=str, default="data/train_bdd100k", help="Output custom dataset dir.")
    parser.add_argument("--labels-json", type=str, default="", help="Optional BDD labels JSON with attributes.")
    parser.add_argument("--default-mu", type=float, default=0.75, help="Default mu value for all rows.")
    parser.add_argument("--default-surface-type", type=str, default="bdd_unknown", help="Default surface type label.")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Target frame sampling rate.")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="Limit frames per video (0 = no limit).")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize width (0 keeps source width).")
    parser.add_argument("--resize-height", type=int, default=0, help="Resize height (0 keeps source height).")
    parser.add_argument("--clear-output", action="store_true", help="Clear output folder before writing.")

    args = parser.parse_args()

    if not args.video_file and not args.video_dir:
        parser.error("Provide at least one of --video-file or --video-dir")

    convert_bdd_to_custom(
        video_file=args.video_file,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        labels_json=args.labels_json,
        default_mu=args.default_mu,
        default_surface_type=args.default_surface_type,
        sample_fps=args.sample_fps,
        max_frames_per_video=args.max_frames_per_video,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        clear_output=args.clear_output,
    )


if __name__ == "__main__":
    main()
