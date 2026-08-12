import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class MultiModalBrakingDataset(Dataset):
    """Legacy custom dataset adapter for synchronized image + CAN + friction data."""

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
    PREFERRED_CAN_COLUMNS = [
        "a_x", "a_y",
        "omega_FL", "omega_FR", "omega_RL", "omega_RR",
        "steering_angle", "brake_pressure", "v_x", "yaw_rate",
        "tire_temp_FL", "tire_temp_FR",
    ]
    TARGET_COLUMNS = ["mu", "target_mu", "friction", "friction_coeff", "friction_coefficient"]

    def __init__(
        self,
        data_dir,
        can_file: str = "can_data.csv",
        image_dir: str = "images",
        seq_len: int = 50,
        transform=None,
        train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = max(1, int(seq_len))
        self.transform = transform
        self.train = train

        can_path = self.data_dir / can_file
        if not can_path.exists():
            raise FileNotFoundError(f"CAN file not found: {can_path}")
        self.can_data = pd.read_csv(can_path)
        if self.can_data.empty:
            raise ValueError(f"CAN file is empty: {can_path}")

        self.image_dir = self.data_dir / image_dir
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_files = sorted(
            [f.name for f in self.image_dir.iterdir() if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS]
        )
        if not self.image_files:
            raise ValueError(f"No images found in: {self.image_dir}")

        self.can_feature_columns = self._resolve_can_feature_columns()

        # Public attribute used by scripts/preprocess_data.py
        self.synchronized_data = self._synchronize_data()
        self.sequences = self._create_sequences()

    def _resolve_can_feature_columns(self) -> List[str]:
        """Resolve CAN feature columns robustly with preference order and safe fallback."""
        available = set(self.can_data.columns)
        preferred = [c for c in self.PREFERRED_CAN_COLUMNS if c in available]
        if preferred:
            return preferred

        blocked = {"timestamp", *self.TARGET_COLUMNS}
        numeric = [
            c for c in self.can_data.columns
            if c not in blocked and pd.api.types.is_numeric_dtype(self.can_data[c])
        ]
        return numeric

    @staticmethod
    def _extract_timestamp_from_filename(img_file: str) -> Optional[float]:
        """Extract timestamp from filename using numeric prefixes/tokens."""
        stem = Path(img_file).stem
        lead = re.match(r"^(-?\d+(?:\.\d+)?)", stem)
        if lead:
            return float(lead.group(1))
        any_num = re.search(r"(-?\d+(?:\.\d+)?)", stem)
        if any_num:
            return float(any_num.group(1))
        return None

    def _normalize_can_timestamps(self) -> Optional[np.ndarray]:
        """Normalize CAN timestamps to numeric array when possible."""
        if "timestamp" not in self.can_data.columns:
            return None

        ts = pd.to_numeric(self.can_data["timestamp"], errors="coerce")
        ts_arr = np.asarray(ts, dtype=np.float64)
        if np.isnan(ts_arr).all():
            ts_dt = pd.to_datetime(self.can_data["timestamp"], errors="coerce")
            ts_ns = pd.to_numeric(ts_dt, errors="coerce")
            ts_arr = np.asarray(ts_ns, dtype=np.float64) / 1e9
            if np.isnan(ts_arr).all():
                return None
        return ts_arr

    def _synchronize_data(self):
        """Synchronize image frames with nearest CAN rows by timestamp; fallback to index-order alignment."""
        synchronized = []
        can_ts = self._normalize_can_timestamps()

        for i, img_file in enumerate(self.image_files):
            img_path = str(self.image_dir / img_file)
            ts = self._extract_timestamp_from_filename(img_file)

            if can_ts is not None and ts is not None:
                closest_idx = int(np.abs(can_ts - ts).argmin())
                timestamp = float(ts)
            else:
                closest_idx = min(i, len(self.can_data) - 1)
                timestamp = float(closest_idx)

            synchronized.append({
                "image_path": img_path,
                "timestamp": timestamp,
                "can_data": self.can_data.iloc[closest_idx].to_dict(),
            })

        return synchronized

    def _extract_target_mu(self, can_dict: Dict) -> float:
        for key in self.TARGET_COLUMNS:
            if key in can_dict and pd.notna(can_dict[key]):
                return float(can_dict[key])
        return 0.5

    def _extract_can_features(self, can_dict: Dict) -> List[float]:
        if not self.can_feature_columns:
            return [0.0]
        return [float(can_dict.get(col, 0.0)) for col in self.can_feature_columns]

    def _create_sequences(self):
        """Create fixed-length CAN sequences aligned to terminal image frame."""
        sequences = []
        if len(self.synchronized_data) < self.seq_len:
            return sequences

        for i in range(len(self.synchronized_data) - self.seq_len + 1):
            seq_can_data = []
            for j in range(i, i + self.seq_len):
                can_dict = self.synchronized_data[j]["can_data"]
                seq_can_data.append(self._extract_can_features(can_dict))

            end = self.synchronized_data[i + self.seq_len - 1]
            sequences.append({
                "image_path": end["image_path"],
                "can_sequence": np.asarray(seq_can_data, dtype=np.float32),
                "target_mu": self._extract_target_mu(end["can_data"]),
            })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        image = Image.open(seq["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        can_sequence = torch.tensor(seq["can_sequence"], dtype=torch.float32)
        target_mu = torch.tensor([seq["target_mu"]], dtype=torch.float32)

        return {
            "image": image,
            "can_sequence": can_sequence,
            "target_mu": target_mu,
        }


def get_dataloaders(data_dir, batch_size: int = 32, seq_len: int = 50, val_dir: Optional[str] = None):
    """Get train and validation dataloaders for custom synchronized data."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultiModalBrakingDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        transform=transform,
        train=True,
    )

    candidate_val_dir = val_dir
    if candidate_val_dir is None:
        replaced = str(data_dir).replace("train", "val")
        if replaced != str(data_dir) and os.path.exists(replaced):
            candidate_val_dir = replaced

    if candidate_val_dir and os.path.exists(candidate_val_dir):
        val_dataset = MultiModalBrakingDataset(
            data_dir=candidate_val_dir,
            seq_len=seq_len,
            transform=transform,
            train=False,
        )
    else:
        val_len = max(1, int(0.2 * len(train_dataset)))
        train_len = max(1, len(train_dataset) - val_len)
        train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader