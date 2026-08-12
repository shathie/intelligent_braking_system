"""
Integration for external datasets:
1. Tsinghua University Road Surface Dataset
2. Mendeley Multi-Modal Vehicle Dataset
"""

import os
import csv
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional, Tuple
from utils.preprocessing import ImagePreprocessor, CANSignalPreprocessor, PreprocessingConfig


ROAD_SURFACE_CLASS_ORDER = [
    'dry_asphalt_severe', 'dry_asphalt_slight', 'dry_asphalt_smooth',
    'dry_concrete_severe', 'dry_concrete_slight', 'dry_concrete_smooth',
    'dry_gravel', 'dry_mud', 'fresh_snow', 'ice', 'melted_snow',
    'water_asphalt_severe', 'water_asphalt_slight', 'water_asphalt_smooth',
    'water_concrete_severe', 'water_concrete_slight', 'water_concrete_smooth',
    'water_gravel', 'water_mud',
    'wet_asphalt_severe', 'wet_asphalt_slight', 'wet_asphalt_smooth',
    'wet_concrete_severe', 'wet_concrete_slight', 'wet_concrete_smooth',
    'wet_gravel', 'wet_mud',
]

ROAD_SURFACE_CLASS_TO_ID = {name: i for i, name in enumerate(ROAD_SURFACE_CLASS_ORDER)}

# Rare classes that have near-zero representation across all available datasets.
# Map them to the nearest asphalt equivalent so the model doesn't waste capacity
# on classes it can never learn. Remapping preserves the friction regime.
RARE_CLASS_ALIASES: Dict[str, str] = {
    'water_concrete_severe':  'water_asphalt_severe',
    'water_concrete_slight':  'water_asphalt_slight',
    'water_concrete_smooth':  'water_asphalt_smooth',
    'wet_concrete_severe':    'wet_asphalt_severe',
    'wet_concrete_slight':    'wet_asphalt_slight',
    'wet_concrete_smooth':    'wet_asphalt_smooth',
    'water_asphalt_slight':   'water_asphalt_smooth',  # also under-represented
}

ROAD_SURFACE_MU_PRIORS = {
    'dry_asphalt_severe': 0.90, 'dry_asphalt_slight': 0.85, 'dry_asphalt_smooth': 0.80,
    'dry_concrete_severe': 0.88, 'dry_concrete_slight': 0.83, 'dry_concrete_smooth': 0.78,
    'dry_gravel': 0.65, 'dry_mud': 0.55,
    'fresh_snow': 0.20, 'ice': 0.10, 'melted_snow': 0.25,
    'water_asphalt_severe': 0.45, 'water_asphalt_slight': 0.50, 'water_asphalt_smooth': 0.55,
    'water_concrete_severe': 0.42, 'water_concrete_slight': 0.47, 'water_concrete_smooth': 0.52,
    'water_gravel': 0.35, 'water_mud': 0.30,
    'wet_asphalt_severe': 0.55, 'wet_asphalt_slight': 0.60, 'wet_asphalt_smooth': 0.65,
    'wet_concrete_severe': 0.50, 'wet_concrete_slight': 0.55, 'wet_concrete_smooth': 0.60,
    'wet_gravel': 0.40, 'wet_mud': 0.35,
}


class THURoadSurfaceDataset(Dataset):
    """
    Dataset class for Tsinghua University Road Surface Dataset.
    
    Assumes the dataset has:
    - Directory structure: dataset_root/{surface_type}/*.jpg
    - Surface types: dry, wet, icy, rough, slippery
    - Optional: CSV file with additional metadata
    """
    
    SURFACE_MAP = {
        'dry': 0,
        'wet': 1,
        'icy': 2,
        'rough': 3,
        'slippery': 4
    }
    
    def __init__(self, root_dir: str, transform=None, config: PreprocessingConfig = None):
        """
        Args:
            root_dir: Path to dataset root directory
            transform: Optional transform to apply to images
            config: Preprocessing configuration
        """
        self.root_dir = root_dir
        self.transform = transform or ImagePreprocessor(config).transform
        self.config = config or PreprocessingConfig()
        
        # Load dataset
        self.samples = self._load_dataset()
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from classic or split-folder THU structures."""
        samples = []

        split_dirs = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
            and (
                d.lower().startswith('train')
                or d.lower().startswith('vali')
                or d.lower().startswith('val')
                or d.lower().startswith('test')
            )
        ]

        class_name_to_dirs: Dict[str, List[str]] = {}

        if split_dirs:
            for split_dir in split_dirs:
                split_path = os.path.join(self.root_dir, split_dir)
                class_dirs = [
                    d for d in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, d))
                ]
                for class_name in class_dirs:
                    class_name_to_dirs.setdefault(class_name, []).append(os.path.join(split_path, class_name))
        else:
            # Backward-compatible flat structure: root/{surface_type}/*.jpg
            for class_name in self.SURFACE_MAP.keys():
                class_path = os.path.join(self.root_dir, class_name)
                if os.path.isdir(class_path):
                    class_name_to_dirs.setdefault(class_name, []).append(class_path)

        if not class_name_to_dirs:
            return samples

        # Map coarse THU class names to the canonical 27-class taxonomy so they
        # are recognised by collate_mixed and are NOT filtered to surface_id=-1.
        _THU_COARSE_MAP: Dict[str, str] = {
            'dry':      'dry_asphalt_smooth',
            'wet':      'wet_asphalt_smooth',
            'icy':      'ice',
            'ice':      'ice',
            'rough':    'dry_asphalt_severe',
            'slippery': 'wet_asphalt_severe',
            'snow':     'fresh_snow',
            'sandy':    'dry_asphalt_slight',
        }

        class_names = sorted(class_name_to_dirs.keys())
        if all(name in ROAD_SURFACE_CLASS_TO_ID for name in class_names):
            class_map = ROAD_SURFACE_CLASS_TO_ID
        else:
            # Remap coarse names then resolve through the canonical taxonomy
            remapped: Dict[str, List[str]] = {}
            for raw_name, dirs in class_name_to_dirs.items():
                canonical = _THU_COARSE_MAP.get(raw_name.lower(), None)
                if canonical and canonical in ROAD_SURFACE_CLASS_TO_ID:
                    key = canonical
                elif raw_name in ROAD_SURFACE_CLASS_TO_ID:
                    key = raw_name
                else:
                    key = raw_name  # keep as-is (will be masked -1 in collate)
                remapped.setdefault(key, []).extend(dirs)
            class_name_to_dirs = remapped
            class_map = {
                name: ROAD_SURFACE_CLASS_TO_ID.get(
                    RARE_CLASS_ALIASES.get(name, name),
                    ROAD_SURFACE_CLASS_TO_ID.get(name, i)
                )
                for i, name in enumerate(sorted(class_name_to_dirs.keys()))
            }

        for class_name, class_dirs in class_name_to_dirs.items():
            class_id = class_map[class_name]
            for class_dir in class_dirs:
                for root, _, files in os.walk(class_dir):
                    for file_name in files:
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(root, file_name)
                            samples.append({
                                'image_path': img_path,
                                'surface_type': class_name,
                                'surface_id': class_id,
                                'mu': self._estimate_mu(class_name)
                            })

        return samples
    
    def _estimate_mu(self, surface_type: str) -> float:
        """Estimate friction coefficient from surface type."""
        name = surface_type.lower()
        if name in ROAD_SURFACE_MU_PRIORS:
            return float(ROAD_SURFACE_MU_PRIORS[name])
        if 'ice' in name:
            return 0.1
        if 'snow' in name:
            return 0.2
        if 'wet' in name or 'water' in name:
            return 0.4
        if 'mud' in name:
            return 0.3
        if 'gravel' in name:
            return 0.5
        if 'rough' in name:
            return 0.6
        if 'dry' in name:
            return 0.8
        if 'slippery' in name:
            return 0.2
        return 0.5
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'surface_id': torch.tensor(sample['surface_id'], dtype=torch.long),
            'surface_type': sample['surface_type'],
            'mu': torch.tensor(sample['mu'], dtype=torch.float32)
        }


class MendeleyVehicleDataset(Dataset):
    """
    Dataset class for Mendeley Multi-Modal Vehicle Dataset.
    
    Assumes the dataset has:
    - CSV file with CAN bus data
    - Image directory with synchronized images
    - Column 'timestamp' for synchronization
    - Column 'mu' or 'friction' for ground truth
    """
    
    def __init__(self, data_dir: str, can_file: str = 'can_data.csv',
                 image_dir: str = 'images', transform=None,
                 config: PreprocessingConfig = None, seq_length: int = 50,
                 strict_real_data: bool = True):
        """
        Args:
            data_dir: Path to dataset directory
            can_file: Name of CAN data CSV file
            image_dir: Name of image directory
            transform: Optional transform for images
            config: Preprocessing configuration
            seq_length: Length of CAN sequences
        """
        self.data_dir = data_dir
        self.can_file = can_file
        self.image_dir = image_dir
        self.transform = transform or ImagePreprocessor(config).transform
        self.config = config or PreprocessingConfig()
        self.seq_length = seq_length
        self.strict_real_data = strict_real_data
        
        # Load data
        self.can_data, self.image_paths = self._load_data()
        self.synchronized = self._synchronize_data()
    
    def _load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load CAN data and image paths from real dataset files only."""
        can_path = os.path.join(self.data_dir, self.can_file)
        image_dir_path = os.path.join(self.data_dir, self.image_dir)
        if not os.path.exists(can_path):
            raise FileNotFoundError(
                f"Mendeley CAN file not found: {can_path}. "
                "Please place the real dataset in data/external/mendeley_vehicle."
            )

        can_data = pd.read_csv(can_path)
        if can_data.empty:
            raise ValueError(f"Mendeley CAN file is empty: {can_path}")
        if 'timestamp' not in can_data.columns:
            raise ValueError(f"Mendeley CAN file must contain 'timestamp' column: {can_path}")

        # Normalize timestamps to numeric seconds for reliable synchronization.
        ts_numeric = pd.to_numeric(can_data['timestamp'], errors='coerce')
        if ts_numeric.isna().all():
            ts_datetime = pd.to_datetime(can_data['timestamp'], errors='coerce')
            if ts_datetime.isna().all():
                raise ValueError(
                    "Unable to parse Mendeley CAN timestamps as numeric or datetime values."
                )
            ts_numeric = ts_datetime.astype('int64') / 1e9
        can_data['timestamp'] = ts_numeric.astype(float)

        if not os.path.exists(image_dir_path):
            raise FileNotFoundError(
                f"Mendeley image directory not found: {image_dir_path}. "
                "Please place real synchronized images under images/."
            )

        image_paths = [
            os.path.join(image_dir_path, f)
            for f in os.listdir(image_dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        image_paths.sort()
        if not image_paths:
            raise ValueError(f"No images found in Mendeley image directory: {image_dir_path}")

        return can_data, image_paths

    def _extract_timestamp_from_filename(self, image_path: str) -> Optional[float]:
        """Extract timestamp from image filename if present."""
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # Prefer leading numeric token (common for synchronized recordings).
        leading = re.match(r'^(-?\d+(?:\.\d+)?)', stem)
        if leading:
            return float(leading.group(1))

        # Fallback: any numeric token in the stem.
        any_match = re.search(r'(-?\d+(?:\.\d+)?)', stem)
        if any_match:
            return float(any_match.group(1))

        return None
    
    def _synchronize_data(self) -> List[Dict]:
        """Synchronize images and CAN data."""
        synchronized = []

        can_timestamps = self.can_data['timestamp'].values
        time_tolerance = max(1.0 / float(self.config.sample_rate), 0.01)

        for image_path in self.image_paths:
            image_timestamp = self._extract_timestamp_from_filename(image_path)
            if image_timestamp is None:
                continue

            time_diff = np.abs(can_timestamps - image_timestamp)
            closest_idx = int(np.argmin(time_diff))
            if time_diff[closest_idx] <= time_tolerance:
                synchronized.append({
                    'image_path': image_path,
                    'timestamp': float(image_timestamp),
                    'can_data': self.can_data.iloc[closest_idx].to_dict()
                })

        # If filenames do not encode timestamps, align by index order.
        if not synchronized:
            limit = min(len(self.image_paths), len(self.can_data))
            for idx in range(limit):
                synchronized.append({
                    'image_path': self.image_paths[idx],
                    'timestamp': float(self.can_data.iloc[idx]['timestamp']),
                    'can_data': self.can_data.iloc[idx].to_dict()
                })

        if not synchronized and self.strict_real_data:
            raise ValueError(
                "Failed to synchronize Mendeley images with CAN data. "
                "Verify timestamp format or ordering between can_data.csv and images/."
            )

        return synchronized
    
    def __len__(self) -> int:
        if not self.synchronized:
            return 0
        return max(0, len(self.synchronized) - self.seq_length + 1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from utils.preprocessing import CANSignalPreprocessor

        if not self.synchronized:
            raise IndexError("Dataset is empty")

        end_idx = idx + self.seq_length - 1
        if end_idx >= len(self.synchronized):
            end_idx = len(self.synchronized) - 1

        end_sample = self.synchronized[end_idx]

        if not os.path.exists(end_sample['image_path']):
            raise FileNotFoundError(f"Missing Mendeley image file: {end_sample['image_path']}")
        image = Image.open(end_sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        can_preprocessor = CANSignalPreprocessor(self.config)
        can_sequence = []
        for i in range(max(0, idx), min(len(self.synchronized), idx + self.seq_length)):
            sample = self.synchronized[i]
            can_frame = can_preprocessor.preprocess_frame(sample['can_data'])
            can_sequence.append(can_frame)

        if len(can_sequence) < self.seq_length:
            while len(can_sequence) < self.seq_length:
                can_sequence.append(can_sequence[-1] if can_sequence else np.zeros(len(self.config.can_signals), dtype=np.float32))

        can_sequence = torch.tensor(np.array(can_sequence), dtype=torch.float32)

        # Get target friction
        target_mu = end_sample['can_data'].get('mu', None)
        if target_mu is None:
            # Try alternative column names
            for col in ['friction', 'friction_coeff', 'friction_coefficient']:
                if col in end_sample['can_data']:
                    target_mu = end_sample['can_data'][col]
                    break
        
        if target_mu is None:
            raise ValueError(
                "Mendeley sample does not include target friction (mu/friction/friction_coeff)."
            )
        
        return {
            'image': image,
            'can_sequence': can_sequence,
            'target_mu': torch.tensor(float(target_mu), dtype=torch.float32)
        }


class MendeleyRoadSurfaceDataset(Dataset):
    """Image-only Mendeley road-surface dataset using split folders."""

    def __init__(self, root_dir: str, split: str = 'train-set', transform=None,
                 config: PreprocessingConfig = None):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform or ImagePreprocessor(config).transform
        self.config = config or PreprocessingConfig()
        self.samples = self._load_dataset()

    def _estimate_mu(self, label: str) -> float:
        """Estimate friction coefficient from class/condition label text."""
        name = label.lower()
        if name in ROAD_SURFACE_MU_PRIORS:
            return float(ROAD_SURFACE_MU_PRIORS[name])
        if 'ice' in name:
            return 0.1
        if 'snow' in name:
            return 0.2
        if 'wet' in name or 'water' in name:
            return 0.4
        if 'mud' in name:
            return 0.3
        if 'gravel' in name:
            return 0.5
        if 'dry' in name:
            return 0.8
        return 0.5

    def _load_dataset(self) -> List[Dict]:
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Mendeley split directory not found: {self.split_dir}")

        class_dirs = [
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ]
        if not class_dirs:
            raise ValueError(f"No class folders found in Mendeley split: {self.split_dir}")

        class_dirs = sorted(class_dirs)
        if all(class_name in ROAD_SURFACE_CLASS_TO_ID for class_name in class_dirs):
            class_map = ROAD_SURFACE_CLASS_TO_ID
        else:
            class_map = {class_name: i for i, class_name in enumerate(class_dirs)}

        samples = []
        for class_name, class_id in class_map.items():
            class_dir = os.path.join(self.split_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for file_name in files:
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file_name)
                        samples.append({
                            'image_path': image_path,
                            'surface_type': class_name,
                            'surface_id': class_id,
                            'mu': self._estimate_mu(class_name)
                        })

        if not samples:
            raise ValueError(f"No images found in Mendeley split: {self.split_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'surface_id': torch.tensor(sample['surface_id'], dtype=torch.long),
            'surface_type': sample['surface_type'],
            'mu': torch.tensor(sample['mu'], dtype=torch.float32),
            'target_mu': torch.tensor(sample['mu'], dtype=torch.float32)
        }


class DAWNWeatherDataset(Dataset):
    """Image-only DAWN adverse-weather dataset."""

    WEATHER_ORDER = ['Fog', 'Rain', 'Sand', 'Snow']

    def __init__(self, root_dir: str, weather_types: Optional[List[str]] = None,
                 transform=None, config: PreprocessingConfig = None):
        self.root_dir = root_dir
        self.weather_types = weather_types or self.WEATHER_ORDER
        self.transform = transform or ImagePreprocessor(config).transform
        self.config = config or PreprocessingConfig()
        self.samples = self._load_dataset()

    # Map DAWN weather conditions to canonical 27-class surface names so they
    # pass the collate_mixed surface_type check instead of being masked to -1.
    _WEATHER_TO_SURFACE: Dict[str, str] = {
        'fog':  'wet_asphalt_slight',
        'rain': 'wet_asphalt_smooth',
        'sand': 'dry_asphalt_slight',
        'snow': 'fresh_snow',
    }

    def _estimate_mu(self, weather: str) -> float:
        mu_map = {
            'fog': 0.55, 'rain': 0.40,
            'sand': 0.45, 'snow': 0.20,
            # also accept canonical names
            'wet_asphalt_slight': 0.55, 'wet_asphalt_smooth': 0.40,
            'dry_asphalt_slight': 0.85, 'fresh_snow': 0.20,
        }
        return mu_map.get(weather.lower(), 0.5)

    def _load_dataset(self) -> List[Dict]:
        samples = []

        for weather in self.weather_types:
            weather_dir = os.path.join(self.root_dir, weather)
            if not os.path.exists(weather_dir):
                continue

            # Map to canonical 27-class name; fall back to raw lowercase
            canonical = self._WEATHER_TO_SURFACE.get(weather.lower(), weather.lower())
            canonical_resolved = RARE_CLASS_ALIASES.get(canonical, canonical)
            surface_id = ROAD_SURFACE_CLASS_TO_ID.get(canonical_resolved, -1)

            for file_name in os.listdir(weather_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(weather_dir, file_name)
                    samples.append({
                        'image_path': image_path,
                        'surface_type': canonical,
                        'surface_id': surface_id,
                        'mu': self._estimate_mu(weather)
                    })

        if not samples:
            raise ValueError(f"No DAWN images found under: {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'surface_id': torch.tensor(sample['surface_id'], dtype=torch.long),
            'surface_type': sample['surface_type'],
            'mu': torch.tensor(sample['mu'], dtype=torch.float32),
            'target_mu': torch.tensor(sample['mu'], dtype=torch.float32)
        }


class BDD100KDataset(Dataset):
    """
    Image dataset built from the BDD100K collection.

    Supports two layouts:
    1. Pre-extracted frames organised in condition sub-folders:
       ``<root_dir>/<condition>/*.jpg``  (condition = clear / rainy / snowy / foggy / etc.)
    2. A flat collection of frames inside ``<root_dir>/videos/`` (or any sub-path).
    If a BDD100K labels JSON is present at ``<root_dir>/labels.json`` or
    ``<root_dir>/bdd100k_labels_images_*.json`` its weather/scene attributes are
    used for surface-type tagging; otherwise the folder name or filename heuristics
    are applied.
    """

    # Map BDD100K weather/condition strings to friction priors
    _CONDITION_MU: Dict[str, float] = {
        'clear': 0.80, 'partly cloudy': 0.75, 'overcast': 0.70,
        'rainy': 0.40, 'foggy': 0.55, 'snowy': 0.20,
        'undefined': 0.60,
    }

    _CONDITION_SURFACE: Dict[str, str] = {
        'clear': 'dry_asphalt_smooth',
        'partly cloudy': 'dry_asphalt_slight',
        'overcast': 'dry_asphalt_slight',
        'rainy': 'wet_asphalt_smooth',
        'foggy': 'wet_asphalt_slight',
        'snowy': 'fresh_snow',
        'undefined': 'dry_asphalt_smooth',
    }

    def __init__(self, root_dir: str, transform=None,
                 config: 'PreprocessingConfig' = None):
        self.root_dir = root_dir
        self.transform = transform or ImagePreprocessor(config).transform
        self.config = config or PreprocessingConfig()
        self.samples = self._load_dataset()

    # ------------------------------------------------------------------
    def _infer_condition_from_path(self, path: str) -> str:
        """Infer weather condition from directory or file name."""
        lower = path.lower()
        if 'rain' in lower:
            return 'rainy'
        if 'snow' in lower:
            return 'snowy'
        if 'fog' in lower:
            return 'foggy'
        if 'clear' in lower:
            return 'clear'
        if 'cloudy' in lower or 'overcast' in lower:
            return 'overcast'
        return 'clear'

    def _load_labels_json(self) -> Dict[str, str]:
        """Load BDD100K labels JSON → {video_stem: condition} mapping."""
        label_map: Dict[str, str] = {}
        candidates = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith('.json'):
                candidates.append(os.path.join(self.root_dir, fname))
        for candidate in candidates:
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data:
                    name = item.get('name', '')
                    attrs = item.get('attributes', {})
                    weather = attrs.get('weather', 'undefined').lower()
                    stem = os.path.splitext(name)[0]
                    label_map[stem] = weather
                if label_map:
                    break
            except Exception:
                continue
        return label_map

    def _load_dataset(self) -> List[Dict]:
        samples: List[Dict] = []
        label_map = self._load_labels_json()

        # Walk the root directory for images
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(dirpath, fname)
                stem = os.path.splitext(fname)[0]

                # Priority: labels JSON → directory name → filename heuristic
                if stem in label_map:
                    condition = label_map[stem]
                else:
                    rel_dir = os.path.relpath(dirpath, self.root_dir)
                    condition = self._infer_condition_from_path(rel_dir + '/' + fname)

                surface_type = self._CONDITION_SURFACE.get(condition, 'dry_asphalt_smooth')
                mu = self._CONDITION_MU.get(condition, 0.70)
                surface_id = ROAD_SURFACE_CLASS_TO_ID.get(
                    RARE_CLASS_ALIASES.get(surface_type, surface_type), 0)

                samples.append({
                    'image_path': img_path,
                    'surface_type': surface_type,
                    'surface_id': surface_id,
                    'mu': mu,
                })

        if not samples:
            raise ValueError(f"No images found in BDD100K directory: {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'surface_id': torch.tensor(sample['surface_id'], dtype=torch.long),
            'surface_type': sample['surface_type'],
            'mu': torch.tensor(sample['mu'], dtype=torch.float32),
            'target_mu': torch.tensor(sample['mu'], dtype=torch.float32),
        }


class KITTIRawDataset(Dataset):
    """
    Image dataset built from KITTI raw sequences.

    Scans ``<root_dir>/*/image_02/data/`` (preferred) or
    ``<root_dir>/*/image_00/data_rect/`` for camera frames.
    KITTI sequences are typically recorded on dry urban roads, so
    ``dry_asphalt_smooth`` is used as the default surface type.
    """

    def __init__(self, root_dir: str, transform=None,
                 config: 'PreprocessingConfig' = None):
        self.root_dir = root_dir
        self.transform = transform or ImagePreprocessor(config).transform
        self.config = config or PreprocessingConfig()
        self.samples = self._load_dataset()

    _CAMERA_SUBDIRS = [
        'image_02/data', 'image_02/data_rect',
        'image_00/data', 'image_00/data_rect',
        'image_01/data', 'image_01/data_rect',
        'image_03/data',
    ]

    def _load_dataset(self) -> List[Dict]:
        samples: List[Dict] = []
        surface_type = 'dry_asphalt_smooth'
        mu = float(ROAD_SURFACE_MU_PRIORS.get(surface_type, 0.80))
        surface_id = ROAD_SURFACE_CLASS_TO_ID.get(
            RARE_CLASS_ALIASES.get(surface_type, surface_type), 2)

        if not os.path.isdir(self.root_dir):
            raise ValueError(f"KITTI root directory not found: {self.root_dir}")

        for seq_name in sorted(os.listdir(self.root_dir)):
            seq_dir = os.path.join(self.root_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            img_dir = None
            for sub in self._CAMERA_SUBDIRS:
                candidate = os.path.join(seq_dir, sub)
                if os.path.isdir(candidate):
                    img_dir = candidate
                    break

            # Fallback: any sub-directory that starts with 'image_'
            if img_dir is None:
                for sub in os.listdir(seq_dir):
                    if sub.startswith('image_') and os.path.isdir(os.path.join(seq_dir, sub)):
                        for inner in os.listdir(os.path.join(seq_dir, sub)):
                            inner_path = os.path.join(seq_dir, sub, inner)
                            if os.path.isdir(inner_path):
                                img_dir = inner_path
                                break
                    if img_dir:
                        break

            if img_dir is None:
                continue

            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append({
                        'image_path': os.path.join(img_dir, fname),
                        'surface_type': surface_type,
                        'surface_id': surface_id,
                        'mu': mu,
                    })

        if not samples:
            raise ValueError(f"No KITTI images found under: {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'surface_id': torch.tensor(sample['surface_id'], dtype=torch.long),
            'surface_type': sample['surface_type'],
            'mu': torch.tensor(sample['mu'], dtype=torch.float32),
            'target_mu': torch.tensor(sample['mu'], dtype=torch.float32),
        }


class CombinedDataset(Dataset):
    """
    Combine multiple datasets for training.
    Useful when you have both THU and Mendeley datasets.
    """
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None,
                 repeat_factor: int = 1):
        """
        Args:
            datasets: List of datasets to combine
            weights: Sampling weights for each dataset
            repeat_factor: Virtually repeat each sample this many times to
                           increase epoch size without duplicating storage.
        """
        self.datasets = [d for d in datasets if len(d) > 0]
        self.weights = weights or [1.0] * len(datasets)

        if len(self.datasets) == 0:
            self.weights = [1.0]
            self.datasets = [datasets[0]] if datasets else []
        elif len(self.datasets) != len(datasets):
            self.weights = [w for d, w in zip(datasets, self.weights) if len(d) > 0]

        self.repeat_factor = max(1, int(repeat_factor))
        
        # Calculate cumulative weights
        self.cumulative_weights = np.cumsum(self.weights)
        self.total_weight = self.cumulative_weights[-1]
    
    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets) * self.repeat_factor
    
    def __getitem__(self, idx: int) -> Dict:
        if not self.datasets:
            raise IndexError("Combined dataset is empty")

        # Strip repeat offset so the logical index always falls within base size
        base_size = sum(len(d) for d in self.datasets)
        idx = idx % base_size if base_size > 0 else 0

        if len(self.datasets) == 1:
            return self.datasets[0][idx % len(self.datasets[0])]

        # Select dataset based on weights
        r = np.random.rand() * self.total_weight
        dataset_idx = int(np.searchsorted(self.cumulative_weights, r))
        dataset_idx = min(dataset_idx, len(self.datasets) - 1)
        
        # Get local index within the chosen dataset
        local_idx = idx % len(self.datasets[dataset_idx])
        
        return self.datasets[dataset_idx][local_idx]


# Factory function to create dataset from config
def create_dataset(dataset_type: str, data_dir: str, **kwargs) -> Dataset:
    """
    Factory function to create dataset based on type.
    
    Args:
        dataset_type: Type of dataset ('thu', 'mendeley', 'custom')
        data_dir: Path to dataset directory
        **kwargs: Additional arguments for dataset
    
    Returns:
        Dataset instance
    """
    if dataset_type == 'thu':
        return THURoadSurfaceDataset(data_dir, **kwargs)
    elif dataset_type == 'mendeley_surface':
        return MendeleyRoadSurfaceDataset(data_dir, **kwargs)
    elif dataset_type == 'mendeley':
        return MendeleyVehicleDataset(data_dir, **kwargs)
    elif dataset_type == 'dawn':
        return DAWNWeatherDataset(data_dir, **kwargs)
    elif dataset_type == 'bdd100k':
        return BDD100KDataset(data_dir, **kwargs)
    elif dataset_type == 'kitti':
        return KITTIRawDataset(data_dir, **kwargs)
    else:
        # Default to custom dataset
        from .datasets import MultiModalBrakingDataset
        return MultiModalBrakingDataset(data_dir, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example 1: Load THU dataset
    thu_dataset = THURoadSurfaceDataset(
        root_dir="data/external/thu_road_surface",
        transform=None
    )
    print(f"THU Dataset: {len(thu_dataset)} samples")
    
    # Example 2: Load Mendeley dataset
    mendeley_dataset = MendeleyVehicleDataset(
        data_dir="data/external/mendeley_vehicle",
        can_file="vehicle_data.csv",
        image_dir="images",
        seq_length=50
    )
    print(f"Mendeley Dataset: {len(mendeley_dataset)} samples")
    
    # Example 3: Combine datasets
    combined = CombinedDataset([thu_dataset, mendeley_dataset], weights=[1.0, 2.0])
    print(f"Combined Dataset: {len(combined)} samples")