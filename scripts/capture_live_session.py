"""
Capture a live driving session as timestamped images + aligned CAN CSV.

Output layout (compatible with MultiModalBrakingDataset):
  data/live_sessions/<session_name>/
    images/
      <timestamp>_cam0.jpg
    can_data.csv
    metadata.json

Example:
  python scripts/capture_live_session.py --duration 60 --fps 10 --can-interface virtual
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required for live capture. Install with: pip install opencv-python") from exc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.can_bus import CANBusInterface, CANDataLogger, CANSignalDecoder, get_default_signals
from utils.preprocessing import PreprocessingConfig


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture synchronized camera images and CAN snapshots.")
    parser.add_argument("--output-root", type=str, default="data/live_sessions", help="Root folder for captured sessions.")
    parser.add_argument("--session-name", type=str, default="", help="Optional session folder name. Defaults to timestamp.")
    parser.add_argument("--duration", type=float, default=30.0, help="Capture duration in seconds.")
    parser.add_argument("--fps", type=float, default=10.0, help="Camera frame rate.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means unlimited until duration).")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")

    parser.add_argument("--can-interface", type=str, default="virtual", choices=["virtual", "socketcan"], help="CAN source interface.")
    parser.add_argument("--can-channel", type=str, default="can0", help="CAN channel name for socketcan.")
    parser.add_argument("--save-raw-can", action="store_true", help="Also save per-message raw CAN decode rows.")

    parser.add_argument("--mu", type=float, default=np.nan, help="Optional friction label to attach to all rows.")
    parser.add_argument("--surface-type", type=str, default="live_unknown", help="Optional surface label for all rows.")
    return parser


def _prepare_output_dirs(output_root: str, session_name: str) -> Dict[str, Path]:
    if not session_name:
        session_name = time.strftime("%Y%m%d_%H%M%S")

    session_dir = Path(output_root) / session_name
    images_dir = session_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)

    return {
        "session_dir": session_dir,
        "images_dir": images_dir,
        "can_csv": session_dir / "can_data.csv",
        "raw_can_csv": session_dir / "can_raw_messages.csv",
        "meta_json": session_dir / "metadata.json",
    }


def _capture_session(args: argparse.Namespace) -> Path:
    paths = _prepare_output_dirs(args.output_root, args.session_name)

    config = PreprocessingConfig()
    decoder = CANSignalDecoder(signals=get_default_signals())
    can_bus = CANBusInterface(decoder=decoder, interface=args.can_interface, channel=args.can_channel)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}.")

    can_bus.open()
    raw_logger = CANDataLogger(str(paths["raw_can_csv"]), decoder) if args.save_raw_can else None

    frame_interval = 1.0 / max(args.fps, 0.1)
    end_time = time.time() + max(args.duration, 0.0)

    rows: List[Dict[str, float]] = []
    latest_signals: Dict[str, float] = {}

    frame_count = 0
    print(f"[INFO] Capturing to: {paths['session_dir']}")
    print("[INFO] Press Ctrl+C to stop early.")

    try:
        while time.time() < end_time:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

            loop_start = time.time()
            timestamp = time.time()

            # Read camera frame
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Camera frame read failed, skipping frame.")
                continue

            image_name = f"{timestamp:.6f}_cam0.jpg"
            image_path = paths["images_dir"] / image_name
            cv2.imwrite(str(image_path), frame)

            # Pull CAN messages for a short window so latest values are fresh.
            poll_deadline = loop_start + min(frame_interval, 0.1)
            while time.time() < poll_deadline:
                msg = can_bus.read_message(timeout=0.001)
                if msg is None:
                    break
                latest_signals.update(msg.signals)
                if raw_logger is not None:
                    raw_logger.log_message(msg)

            row: Dict[str, float] = {"timestamp": timestamp, "image_file": image_name}
            for signal_name in config.can_signals:
                if signal_name == "timestamp":
                    continue
                row[signal_name] = float(latest_signals.get(signal_name, 0.0))

            if not np.isnan(args.mu):
                row["mu"] = float(args.mu)
            row["surface_type"] = args.surface_type
            rows.append(row)

            frame_count += 1
            if frame_count % 20 == 0:
                print(f"[INFO] Frames captured: {frame_count}")

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Capture stopped by user.")
    finally:
        cap.release()
        can_bus.close()

    df = pd.DataFrame(rows)
    df.to_csv(paths["can_csv"], index=False)

    metadata = {
        "session_dir": str(paths["session_dir"]),
        "frames_captured": int(frame_count),
        "duration_sec": float(args.duration),
        "fps": float(args.fps),
        "can_interface": args.can_interface,
        "can_channel": args.can_channel,
        "camera_index": int(args.camera_index),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Session saved: {paths['session_dir']}")
    print(f"[OK] Image frames: {frame_count}")
    print(f"[OK] CAN rows: {len(df)}")
    print("[NEXT] Run preprocessing on this folder with scripts/preprocess_data.py (custom dataset path).")

    return paths["session_dir"]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    _capture_session(args)


if __name__ == "__main__":
    main()
