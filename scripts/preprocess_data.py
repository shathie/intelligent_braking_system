"""
Data preprocessing script.
Converts raw datasets into processed format for training.
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import MultiModalBrakingDataset
from data.external_datasets import (
    THURoadSurfaceDataset, MendeleyRoadSurfaceDataset, DAWNWeatherDataset,
    BDD100KDataset, KITTIRawDataset,
)
from utils.preprocessing import DataSynchronizer, PreprocessingConfig
from utils.visualization import SystemAnalyzer


def _synthetic_can_from_mu(mu: float, seq_length: int, num_signals: int, rng=None) -> torch.Tensor:
    """Generate a physics-plausible CAN sequence from a friction coefficient.

    17 signals (same order as PreprocessingConfig.can_signals):
      0 speed_kmh, 1 acceleration, 2 brake_pressure, 3-6 wheel_speeds (FL/FR/RL/RR),
      7-10 wheel_accel, 11 yaw_rate, 12 lateral_accel, 13 steering_angle,
      14 throttle, 15 gear, 16 engine_rpm
    Values are normalised to roughly [0, 1] so the network sees meaningful variation.
    """
    if rng is None:
        rng = np.random.default_rng(int(abs(mu) * 1e6) % (2**31))

    # Simulate a braking event: start at cruise speed, decelerate based on mu
    v0 = rng.uniform(60.0, 120.0)          # km/h
    decel = mu * 9.81 / 3.6               # m/s^2 converted to km/h per step
    dt = 0.1                               # 10 Hz
    v = v0
    seq = np.zeros((seq_length, num_signals), dtype=np.float32)
    brake_force = rng.uniform(0.4, 1.0)

    for t in range(seq_length):
        v = max(0.0, v - decel * dt * brake_force * 3.6)
        slip_noise = rng.normal(0, 0.02)
        seq[t, 0]  = v / 120.0                        # speed normalised
        seq[t, 1]  = -decel * brake_force / 9.81      # decel normalised
        seq[t, 2]  = brake_force + rng.normal(0, 0.02)  # brake pressure
        seq[t, 3:7] = v / 120.0 + slip_noise          # wheel speeds
        seq[t, 7:11] = -decel * brake_force / 9.81    # wheel accel
        seq[t, 11] = rng.normal(0, 0.01)              # yaw rate
        seq[t, 12] = rng.normal(0, 0.01)              # lat accel
        seq[t, 13] = rng.normal(0, 0.005)             # steering
        seq[t, 14] = 0.0                               # throttle (braking)
        seq[t, 15] = 2.0 / 8.0                        # gear normalised
        seq[t, 16] = rng.uniform(800, 2500) / 8000.0  # rpm normalised

    seq = np.clip(seq, -1.0, 1.0)
    return torch.from_numpy(seq)


class DataPreprocessor:
    """Preprocess datasets for training."""
    
    def __init__(self, output_dir: str = "data/processed", custom_data_dir: str = "data/train"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PreprocessingConfig()
        self.synchronizer = DataSynchronizer(self.config)
        self.thu_max_samples = int(os.getenv("THU_MAX_SAMPLES", "50000"))
        self.mendeley_max_samples = int(os.getenv("MENDELEY_MAX_SAMPLES", "50000"))
        self.dawn_max_samples = int(os.getenv("DAWN_MAX_SAMPLES", "20000"))
        self.bdd_max_samples = int(os.getenv("BDD_MAX_SAMPLES", "10000"))
        self.kitti_max_samples = int(os.getenv("KITTI_MAX_SAMPLES", "2000"))
        self.custom_data_dir = custom_data_dir
    
    def preprocess_thu_dataset(self, input_dir: str) -> Path:
        """Preprocess Tsinghua dataset."""
        print("\n" + "="*70)
        print("PREPROCESSING TSINGHUA DATASET")
        print("="*70)
        
        # Create output directory
        output_dir = self.output_dir / "thu_processed"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        dataset = THURoadSurfaceDataset(input_dir)
        
        # THU is image-only in this pipeline; create image tensors + dummy CAN sequences.
        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        can_output_dir = output_dir / "can_sequences"
        can_output_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        targets: List[float] = []
        metadata_rows = []
        
        selected_samples = dataset.samples[:self.thu_max_samples]
        if len(selected_samples) < len(dataset.samples):
            print(f"[INFO] Capping THU preprocessing to {len(selected_samples)} samples (set THU_MAX_SAMPLES to change).")

        for i, sample in enumerate(tqdm(selected_samples, desc="Processing THU samples")):
            try:
                image_tensor = self.synchronizer.image_preprocessor.preprocess(sample['image_path'])
                torch.save(image_tensor, img_output_dir / f"{i:06d}.pt")

                mu_val = float(sample.get('mu', 0.5))
                can_seq = _synthetic_can_from_mu(mu_val, self.config.seq_length, len(self.config.can_signals))
                torch.save(can_seq, can_output_dir / f"{i:06d}.pt")

                targets.append(mu_val)
                metadata_rows.append({
                    'index': i,
                    'image_path': sample['image_path'],
                    'surface_type': sample.get('surface_type', 'unknown'),
                    'surface_id': int(sample.get('surface_id', -1)),
                    'mu': mu_val,
                    'timestamp': float(sample.get('timestamp', i)),
                })
                processed_count += 1
                
            except Exception as e:
                print(f"[WARNING] Error processing sample {i}: {e}")
                continue

        # Persist compatibility artifacts used by training/evaluation scripts.
        torch.save(torch.tensor(targets, dtype=torch.float32), output_dir / "targets.pt")
        pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)

        # Keep a minimal CAN CSV for backward compatibility with older scripts.
        can_data = pd.DataFrame(metadata_rows)[['timestamp', 'surface_type', 'mu']] if metadata_rows else pd.DataFrame(
            columns=['timestamp', 'surface_type', 'mu']
        )
        can_data.to_csv(output_dir / "can_data.csv", index=False)
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'thu_road_surface',
                'total_samples': processed_count,
                'total_available_samples': len(dataset.samples),
                'max_samples': self.thu_max_samples,
                'seq_length': self.config.seq_length,
                'can_feature_dim': len(self.config.can_signals),
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"[OK] Preprocessed THU dataset saved to: {output_dir}")
        print(f"   Total samples: {processed_count}")
        
        return output_dir
    
    def preprocess_mendeley_dataset(self, input_dir: str) -> Path:
        """Preprocess Mendeley split dataset (image-only)."""
        print("\n" + "="*70)
        print("PREPROCESSING MENDELEY DATASET")
        print("="*70)
        
        # Create output directory
        output_dir = self.output_dir / "mendeley_processed"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load split dataset from real folders
        train_dataset = MendeleyRoadSurfaceDataset(input_dir, split='train-set')

        # Save images and synthetic CAN sequences for compatibility
        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        can_output_dir = output_dir / "can_sequences"
        can_output_dir.mkdir(parents=True, exist_ok=True)

        targets = []
        metadata_rows = []

        selected_samples = train_dataset.samples[:self.mendeley_max_samples]
        if len(selected_samples) < len(train_dataset.samples):
            print(f"[INFO] Capping Mendeley preprocessing to {len(selected_samples)} samples (set MENDELEY_MAX_SAMPLES to change).")

        for i, sample in enumerate(tqdm(selected_samples, desc="Processing Mendeley train samples")):
            image_tensor = self.synchronizer.image_preprocessor.preprocess(sample['image_path'])
            torch.save(image_tensor, img_output_dir / f"{i:06d}.pt")

            mu_val = float(sample['mu'])
            can_seq = _synthetic_can_from_mu(mu_val, self.config.seq_length, len(self.config.can_signals))
            torch.save(can_seq, can_output_dir / f"{i:06d}.pt")

            targets.append(mu_val)
            metadata_rows.append({
                'index': i,
                'image_path': sample['image_path'],
                'surface_type': sample['surface_type'],
                'surface_id': sample['surface_id'],
                'mu': sample['mu']
            })
        
        # Save targets
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        torch.save(targets_tensor, output_dir / "targets.pt")
        pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'mendeley_vehicle',
                'total_samples': len(targets),
                'total_available_samples': len(train_dataset.samples),
                'seq_length': self.config.seq_length,
                'sample_rate': self.config.sample_rate,
                'source_split': 'train-set',
                'max_samples': self.mendeley_max_samples,
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"[OK] Preprocessed Mendeley dataset saved to: {output_dir}")
        print(f"   Total samples: {len(targets)}")
        print(f"   CAN sequence shape: ({self.config.seq_length}, {len(self.config.can_signals)})")
        
        return output_dir

    def preprocess_dawn_dataset(self, input_dir: str) -> Path:
        """Preprocess DAWN weather dataset (image-only)."""
        print("\n" + "="*70)
        print("PREPROCESSING DAWN DATASET")
        print("="*70)

        output_dir = self.output_dir / "dawn_processed"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = DAWNWeatherDataset(input_dir)

        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        can_output_dir = output_dir / "can_sequences"
        can_output_dir.mkdir(parents=True, exist_ok=True)

        targets = []
        metadata_rows = []
        selected_samples = dataset.samples[:self.dawn_max_samples]
        if len(selected_samples) < len(dataset.samples):
            print(f"[INFO] Capping DAWN preprocessing to {len(selected_samples)} samples (set DAWN_MAX_SAMPLES to change).")

        for i, sample in enumerate(tqdm(selected_samples, desc="Processing DAWN samples")):
            image_tensor = self.synchronizer.image_preprocessor.preprocess(sample['image_path'])
            torch.save(image_tensor, img_output_dir / f"{i:06d}.pt")

            mu_val = float(sample['mu'])
            can_seq = _synthetic_can_from_mu(mu_val, self.config.seq_length, len(self.config.can_signals))
            torch.save(can_seq, can_output_dir / f"{i:06d}.pt")

            targets.append(mu_val)
            metadata_rows.append({
                'index': i,
                'image_path': sample['image_path'],
                'weather_type': sample['surface_type'],
                'weather_id': sample['surface_id'],
                'mu': sample['mu']
            })

        torch.save(torch.tensor(targets, dtype=torch.float32), output_dir / "targets.pt")
        pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)

        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'dawn_weather',
                'total_samples': len(targets),
                'total_available_samples': len(dataset.samples),
                'seq_length': self.config.seq_length,
                'max_samples': self.dawn_max_samples,
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)

        print(f"[OK] Preprocessed DAWN dataset saved to: {output_dir}")
        print(f"   Total samples: {len(targets)}")
        return output_dir
    
    def preprocess_bdd100k_dataset(self, input_dir: str) -> Path:
        """Preprocess BDD100K dataset with synthetic CAN sequences."""
        print("\n" + "="*70)
        print("PREPROCESSING BDD100K DATASET")
        print("="*70)

        output_dir = self.output_dir / "bdd_processed"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = BDD100KDataset(input_dir)

        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        can_output_dir = output_dir / "can_sequences"
        can_output_dir.mkdir(parents=True, exist_ok=True)

        targets: List[float] = []
        metadata_rows = []
        selected_samples = dataset.samples[:self.bdd_max_samples]
        if len(selected_samples) < len(dataset.samples):
            print(f"[INFO] Capping BDD100K preprocessing to {len(selected_samples)} samples (set BDD_MAX_SAMPLES to change).")

        for i, sample in enumerate(tqdm(selected_samples, desc="Processing BDD100K samples")):
            try:
                image_tensor = self.synchronizer.image_preprocessor.preprocess(sample['image_path'])
                torch.save(image_tensor, img_output_dir / f"{i:06d}.pt")

                mu_val = float(sample.get('mu', 0.6))
                can_seq = _synthetic_can_from_mu(mu_val, self.config.seq_length, len(self.config.can_signals))
                torch.save(can_seq, can_output_dir / f"{i:06d}.pt")

                targets.append(mu_val)
                metadata_rows.append({
                    'index': i,
                    'image_path': sample['image_path'],
                    'surface_type': sample.get('surface_type', 'dry_asphalt_smooth'),
                    'surface_id': int(sample.get('surface_id', -1)),
                    'mu': mu_val,
                })
            except Exception as e:
                print(f"[WARNING] Error processing BDD100K sample {i}: {e}")
                continue

        torch.save(torch.tensor(targets, dtype=torch.float32), output_dir / "targets.pt")
        pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'bdd100k',
                'total_samples': len(targets),
                'total_available_samples': len(dataset.samples),
                'max_samples': self.bdd_max_samples,
                'seq_length': self.config.seq_length,
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)

        print(f"[OK] Preprocessed BDD100K dataset saved to: {output_dir}")
        print(f"   Total samples: {len(targets)}")
        return output_dir

    def preprocess_kitti_dataset(self, input_dir: str) -> Path:
        """Preprocess KITTI raw dataset with synthetic CAN sequences."""
        print("\n" + "="*70)
        print("PREPROCESSING KITTI DATASET")
        print("="*70)

        output_dir = self.output_dir / "kitti_processed"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = KITTIRawDataset(input_dir)

        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        can_output_dir = output_dir / "can_sequences"
        can_output_dir.mkdir(parents=True, exist_ok=True)

        targets: List[float] = []
        metadata_rows = []
        selected_samples = dataset.samples[:self.kitti_max_samples]
        if len(selected_samples) < len(dataset.samples):
            print(f"[INFO] Capping KITTI preprocessing to {len(selected_samples)} samples (set KITTI_MAX_SAMPLES to change).")

        for i, sample in enumerate(tqdm(selected_samples, desc="Processing KITTI samples")):
            try:
                image_tensor = self.synchronizer.image_preprocessor.preprocess(sample['image_path'])
                torch.save(image_tensor, img_output_dir / f"{i:06d}.pt")

                mu_val = float(sample.get('mu', 0.8))
                can_seq = _synthetic_can_from_mu(mu_val, self.config.seq_length, len(self.config.can_signals))
                torch.save(can_seq, can_output_dir / f"{i:06d}.pt")

                targets.append(mu_val)
                metadata_rows.append({
                    'index': i,
                    'image_path': sample['image_path'],
                    'surface_type': sample.get('surface_type', 'dry_asphalt_smooth'),
                    'surface_id': int(sample.get('surface_id', -1)),
                    'mu': mu_val,
                })
            except Exception as e:
                print(f"[WARNING] Error processing KITTI sample {i}: {e}")
                continue

        torch.save(torch.tensor(targets, dtype=torch.float32), output_dir / "targets.pt")
        pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'kitti_raw',
                'total_samples': len(targets),
                'total_available_samples': len(dataset.samples),
                'max_samples': self.kitti_max_samples,
                'seq_length': self.config.seq_length,
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)

        print(f"[OK] Preprocessed KITTI dataset saved to: {output_dir}")
        print(f"   Total samples: {len(targets)}")
        return output_dir

    def preprocess_custom_dataset(self, input_dir: str) -> Path:
        """Preprocess custom dataset."""
        print("\n" + "="*70)
        print("PREPROCESSING CUSTOM DATASET")
        print("="*70)
        
        # Create output directory
        output_dir = self.output_dir / "custom_processed"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        dataset = MultiModalBrakingDataset(input_dir)

        if len(dataset.synchronized_data) == 0:
            raise ValueError(f"No synchronized samples found in custom dataset: {input_dir}")
        
        # Process data
        effective_seq_len = min(self.config.seq_length, len(dataset.synchronized_data))
        if effective_seq_len < self.config.seq_length:
            print(
                f"[INFO] Reducing custom sequence length from {self.config.seq_length} "
                f"to {effective_seq_len} for small dataset smoke run."
            )
        processed_data = self.synchronizer.create_dataset(dataset.synchronized_data, seq_length=effective_seq_len)
        
        # Save images
        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_tensor in enumerate(tqdm(processed_data['images'], desc="Saving images")):
            img_path = img_output_dir / f"{i:06d}.pt"
            torch.save(img_tensor, img_path)
        
        # Save CAN sequences
        can_output_dir = output_dir / "can_sequences"
        can_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, can_seq in enumerate(tqdm(processed_data['can_sequences'], desc="Saving CAN sequences")):
            can_path = can_output_dir / f"{i:06d}.pt"
            torch.save(can_seq, can_path)
        
        # Save targets
        targets_path = output_dir / "targets.pt"
        torch.save(processed_data['targets'], targets_path)
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'custom',
                'total_samples': len(processed_data['images']),
                'seq_length': effective_seq_len,
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"[OK]Preprocessed custom dataset saved to: {output_dir}")
        print(f"   Total samples: {len(processed_data['images'])}")
        
        return output_dir
    
    def preprocess_all(self, only_custom: bool = False):
        """Preprocess all available datasets."""
        processed_dirs = []

        if only_custom:
            if os.path.exists(self.custom_data_dir):
                processed_dirs.append(self.preprocess_custom_dataset(self.custom_data_dir))
            else:
                print(f"[WARNING] Custom dataset not found at: {os.path.abspath(self.custom_data_dir)}")
            return processed_dirs
        
        # THU dataset
        thu_dir = "data/external/thu_road_surface"
        if os.path.exists(thu_dir):
            processed_dirs.append(self.preprocess_thu_dataset(thu_dir))
        else:
            print(f"[WARNING] THU dataset not found at: {os.path.abspath(thu_dir)}")
        
        # Mendeley dataset
        mendeley_dir = "data/external/mendeley_vehicle"
        if os.path.exists(mendeley_dir):
            processed_dirs.append(self.preprocess_mendeley_dataset(mendeley_dir))
        else:
            print(f"[WARNING] Mendeley dataset not found at: {os.path.abspath(mendeley_dir)}")

        # DAWN dataset
        dawn_dir = "data/external/dawn"
        if os.path.exists(dawn_dir):
            processed_dirs.append(self.preprocess_dawn_dataset(dawn_dir))
        else:
            print(f"[WARNING] DAWN dataset not found at: {os.path.abspath(dawn_dir)}")
        
        # BDD100K dataset
        bdd_dir = "data/external/bdd100k"
        if os.path.exists(bdd_dir):
            processed_dirs.append(self.preprocess_bdd100k_dataset(bdd_dir))
        else:
            print(f"[INFO] BDD100K dataset not found at: {os.path.abspath(bdd_dir)}")

        # KITTI dataset
        kitti_dir = "data/external/kitti_raw"
        if os.path.exists(kitti_dir):
            processed_dirs.append(self.preprocess_kitti_dataset(kitti_dir))
        else:
            print(f"[INFO] KITTI dataset not found at: {os.path.abspath(kitti_dir)}")

        # Custom dataset
        custom_dir = self.custom_data_dir
        if os.path.exists(custom_dir):
            processed_dirs.append(self.preprocess_custom_dataset(custom_dir))
        else:
            print(f"[INFO] Custom dataset not found at: {os.path.abspath(custom_dir)}")
        
        return processed_dirs


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for training.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed tensors.")
    parser.add_argument("--custom-dir", type=str, default=os.getenv("IBS_CUSTOM_DATA_DIR", "data/train"), help="Custom dataset directory with images/ and can_data.csv.")
    parser.add_argument("--only-custom", action="store_true", help="Only preprocess the custom dataset path.")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(output_dir=args.output_dir, custom_data_dir=args.custom_dir)
    processed_dirs = preprocessor.preprocess_all(only_custom=args.only_custom)
    
    print("\n" + "="*70)
    print("[OK] Data preprocessing complete!")
    print(f"Processed datasets saved to: {preprocessor.output_dir}")
    for d in processed_dirs:
        print(f"  - {d}")
    print("="*70)


if __name__ == "__main__":
    main()