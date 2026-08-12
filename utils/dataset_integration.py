"""
Integration for external datasets:
1. Tsinghua University Road Surface Dataset
2. Mendeley Multi-Modal Vehicle Dataset
"""

import os
import csv
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class THURoadSurfaceDataset(Dataset):
    """
    Dataset class for Tsinghua University Road Surface Dataset.
   
    SURFACE_MAP with 26 granular categories matching your requirements.
    """

    SURFACE_MAP = {
        'dry_asphalt_severe': 0, 'dry_asphalt_slight': 1, 'dry_asphalt_smooth': 2,
        'dry_concrete_severe': 3, 'dry_concrete_slight': 4, 'dry_concrete_smooth': 5,
        'dry_gravel': 6, 'dry_mud': 7, 'fresh_snow': 8, 'ice': 9,
        'melted_snow': 10, 'water_asphalt_severe': 11, 'water_asphalt_slight': 12,
        'water_asphalt_smooth': 13, 'water_concrete_severe': 14, 'water_concrete_slight': 15,
        'water_concrete_smooth': 16, 'water_gravel': 17, 'water_mud': 18,
        'wet_asphalt_severe': 19, 'wet_asphalt_slight': 20, 'wet_asphalt_smooth': 21,
        'wet_concrete_severe': 22, 'wet_concrete_slight': 23, 'wet_concrete_smooth': 24,
        'wet_gravel': 25, 'wet_mud': 26
    }

    MU_ESTIMATES = {
        'dry_asphalt_severe': 0.90, 'dry_asphalt_slight': 0.85, 'dry_asphalt_smooth': 0.80,
        'dry_concrete_severe': 0.88, 'dry_concrete_slight': 0.83, 'dry_concrete_smooth': 0.78,
        'dry_gravel': 0.65, 'dry_mud': 0.55,
        'fresh_snow': 0.20, 'ice': 0.10, 'melted_snow': 0.25,
        'water_asphalt_severe': 0.45, 'water_asphalt_slight': 0.50, 'water_asphalt_smooth': 0.55,
        'water_concrete_severe': 0.42, 'water_concrete_slight': 0.47, 'water_concrete_smooth': 0.52,
        'water_gravel': 0.35, 'water_mud': 0.30,
        'wet_asphalt_severe': 0.55, 'wet_asphalt_slight': 0.60, 'wet_asphalt_smooth': 0.65,
        'wet_concrete_severe': 0.50, 'wet_concrete_slight': 0.55, 'wet_concrete_smooth': 0.60,
        'wet_gravel': 0.40, 'wet_mud': 0.35
    }

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        samples = []
        for surface_type, surface_id in self.SURFACE_MAP.items():
            surface_dir = self.root_dir / surface_type
            if not surface_dir.exists():
                continue
            for img_file in surface_dir.glob('*.jpg'):
                samples.append({
                    'image_path': str(img_file),
                    'surface_type': surface_type,
                    'surface_id': surface_id,
                    'mu': self.MU_ESTIMATES.get(surface_type, 0.5)
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)
        return {
            'image': image,
            'surface_id': torch.tensor(sample['surface_id'], dtype=torch.long),
            'surface_type': sample['surface_type'],
            'mu': torch.tensor(sample['mu'], dtype=torch.float32)
        }

class MendeleyVehicleDataset(Dataset):
    """
    Dataset class for Mendeley Multi-Modal Vehicle Dataset.
    """

    def __init__(self, data_dir: str, can_file: str = 'can_data.csv',
                 image_dir: str = 'images', transform=None, seq_length: int = 50):
        self.data_dir = Path(data_dir)
        self.can_file = can_file
        self.image_dir = image_dir
        self.transform = transform
        self.seq_length = seq_length
        self.can_data, self.image_paths = self._load_data()

    def _load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        can_path = self.data_dir / self.can_file
        can_data = pd.read_csv(can_path)
        image_dir_path = self.data_dir / self.image_dir
        image_paths = [str(p) for p in image_dir_path.glob('*.jpg')] if image_dir_path.exists() else []
        return can_data, image_paths

    def __len__(self) -> int:
        return max(len(self.can_data) - self.seq_length + 1, 0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        can_sequence = self.can_data.iloc[idx:idx+self.seq_length].drop(columns=['timestamp']).values
        can_sequence = torch.FloatTensor(can_sequence).T
        img_path = self.image_paths[idx] if idx < len(self.image_paths) else self.image_paths[0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)
        mu = self.can_data.iloc[idx].get('mu', 0.5)
        return {
            'image': image,
            'can_sequence': can_sequence,
            'mu': torch.tensor(float(mu), dtype=torch.float32)
        }