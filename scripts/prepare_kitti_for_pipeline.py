"""
Convert KITTI raw sequence into the project's custom synchronized format.

Input (KITTI raw sequence):
  <kitti_seq>/image_02/data/*.png
  <kitti_seq>/image_02/timestamps.txt
  <kitti_seq>/oxts/data/*.txt
  <kitti_seq>/oxts/timestamps.txt

Output (custom dataset directory):
  <output_dir>/images/<unix_ts>_cam0.jpg
  <output_dir>/can_data.csv

This output is compatible with MultiModalBrakingDataset and scripts/preprocess_data.py --only-custom.
"""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _parse_kitti_timestamp(ts_text: str) -> float:
    ts_text = ts_text.strip().replace("Z", "")
    if not ts_text:
        return 0.0

    # KITTI timestamps usually include nanoseconds; Python datetime handles up to microseconds.
    if "." in ts_text:
        left, right = ts_text.split(".", 1)
        right = (right + "000000")[:6]
        ts_text = f"{left}.{right}"

    dt = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def _read_timestamps(ts_file: Path) -> List[float]:
    if not ts_file.exists():
        return []
    with open(ts_file, "r", encoding="utf-8") as f:
        return [_parse_kitti_timestamp(line) for line in f if line.strip()]


def _read_oxts_row(oxts_file: Path) -> Optional[List[float]]:
    if not oxts_file.exists():
        return None
    with open(oxts_file, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return None
    try:
        return [float(v) for v in line.split()]
    except ValueError:
        return None


def _wheel_omega_from_speed(vx: float, tire_radius_m: float) -> float:
    radius = max(tire_radius_m, 1e-3)
    return float(vx / radius)


def _resolve_kitti_camera(seq_dir: Path, preferred_camera: str) -> Tuple[str, Path, Path]:
    candidates = [preferred_camera, "image_02", "image_03", "image_00", "image_01"]
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    for cam in ordered:
        cam_dir = seq_dir / cam
        img_dir = cam_dir / "data"
        if not img_dir.exists():
            img_dir = cam_dir / "data_rect"
        ts_file = seq_dir / cam / "timestamps.txt"
        if img_dir.exists():
            return cam, img_dir, ts_file

    raise FileNotFoundError(f"No KITTI camera folder found under: {seq_dir}")


def convert_kitti_sequence(
    kitti_seq_dir: str,
    output_dir: str,
    image_camera: str = "image_02",
    image_ext_out: str = ".jpg",
    resize_width: int = 0,
    resize_height: int = 0,
    default_mu: float = 0.85,
    default_surface_type: str = "kitti_dry_road",
    tire_radius_m: float = 0.31,
    max_frames: int = 0,
    clear_output: bool = False,
) -> None:
    seq_dir = Path(kitti_seq_dir)
    out_dir = Path(output_dir)
    images_out_dir = out_dir / "images"

    selected_camera, image_dir, image_ts_file = _resolve_kitti_camera(seq_dir, image_camera)
    oxts_dir = seq_dir / "oxts" / "data"
    oxts_ts_file = seq_dir / "oxts" / "timestamps.txt"

    has_oxts = oxts_dir.exists()

    if clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    images_out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    oxts_files = sorted([p for p in oxts_dir.iterdir() if p.suffix.lower() == ".txt"]) if has_oxts else []

    if not image_files:
        raise ValueError(f"No image files found in: {image_dir}")

    image_timestamps = _read_timestamps(image_ts_file)
    oxts_timestamps = _read_timestamps(oxts_ts_file) if has_oxts else []

    frame_count = min(len(image_files), len(oxts_files)) if has_oxts and oxts_files else len(image_files)
    if max_frames > 0:
        frame_count = min(frame_count, max_frames)

    print(f"[INFO] Selected camera: {selected_camera}")
    if has_oxts and oxts_files:
        print(f"[INFO] Frames available: images={len(image_files)}, oxts={len(oxts_files)}, using={frame_count}")
    else:
        print(f"[INFO] Frames available: images={len(image_files)}, oxts=not_found, using={frame_count} (image-only fallback)")

    csv_path = out_dir / "can_data.csv"
    fieldnames = [
        "timestamp",
        "a_x",
        "a_y",
        "omega_FL",
        "omega_FR",
        "omega_RL",
        "omega_RR",
        "steering_angle",
        "brake_pressure",
        "v_x",
        "yaw_rate",
        "tire_temp_FL",
        "tire_temp_FR",
        "mu",
        "surface_type",
    ]

    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(frame_count):
            if has_oxts and i < len(oxts_files):
                oxts_vals = _read_oxts_row(oxts_files[i])
            else:
                oxts_vals = None

            if oxts_vals is not None and len(oxts_vals) >= 20:
                # KITTI raw OXTS fields used here:
                # vf index 8, ax index 11, ay index 12, wz index 19.
                v_x = float(oxts_vals[8])
                a_x = float(oxts_vals[11])
                a_y = float(oxts_vals[12])
                yaw_rate = float(oxts_vals[19])
            else:
                v_x = 0.0
                a_x = 0.0
                a_y = 0.0
                yaw_rate = 0.0

            omega = _wheel_omega_from_speed(v_x, tire_radius_m)

            if i < len(image_timestamps):
                ts = float(image_timestamps[i])
            elif i < len(oxts_timestamps):
                ts = float(oxts_timestamps[i])
            else:
                ts = float(i) * 0.1

            image = cv2.imread(str(image_files[i]))
            if image is None:
                continue

            if resize_width > 0 and resize_height > 0:
                image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

            out_name = f"{ts:.6f}_cam0{image_ext_out}"
            out_image_path = images_out_dir / out_name
            cv2.imwrite(str(out_image_path), image)

            writer.writerow(
                {
                    "timestamp": ts,
                    "a_x": a_x,
                    "a_y": a_y,
                    "omega_FL": omega,
                    "omega_FR": omega,
                    "omega_RL": omega,
                    "omega_RR": omega,
                    "steering_angle": 0.0,
                    "brake_pressure": 0.0,
                    "v_x": v_x,
                    "yaw_rate": yaw_rate,
                    "tire_temp_FL": 0.0,
                    "tire_temp_FR": 0.0,
                    "mu": float(default_mu),
                    "surface_type": default_surface_type,
                }
            )
            rows_written += 1

    print(f"[OK] Wrote images: {rows_written} to {images_out_dir}")
    print(f"[OK] Wrote CAN CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KITTI raw sequence for project custom preprocessing.")
    parser.add_argument("--kitti-seq-dir", type=str, required=True, help="KITTI raw sequence folder path.")
    parser.add_argument("--output-dir", type=str, default="data/train", help="Output custom dataset dir.")
    parser.add_argument("--image-camera", type=str, default="image_02", help="Preferred camera folder (auto-fallback to available KITTI cameras).")
    parser.add_argument("--image-ext-out", type=str, default=".jpg", choices=[".jpg", ".png"], help="Output image extension.")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize width, 0 keeps original.")
    parser.add_argument("--resize-height", type=int, default=0, help="Resize height, 0 keeps original.")
    parser.add_argument("--default-mu", type=float, default=0.85, help="Default mu label for all frames.")
    parser.add_argument("--surface-type", type=str, default="kitti_dry_road", help="Surface type label for all frames.")
    parser.add_argument("--tire-radius-m", type=float, default=0.31, help="Tire radius for omega estimate.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit converted frames for quick smoke runs (0 = no limit).")
    parser.add_argument("--clear-output", action="store_true", help="Clear output folder before writing.")

    args = parser.parse_args()
    convert_kitti_sequence(
        kitti_seq_dir=args.kitti_seq_dir,
        output_dir=args.output_dir,
        image_camera=args.image_camera,
        image_ext_out=args.image_ext_out,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        default_mu=args.default_mu,
        default_surface_type=args.surface_type,
        tire_radius_m=args.tire_radius_m,
        max_frames=args.max_frames,
        clear_output=args.clear_output,
    )


if __name__ == "__main__":
    main()
