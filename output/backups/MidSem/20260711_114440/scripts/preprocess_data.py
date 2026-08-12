"""
Data preprocessing script.
Converts raw datasets into processed format for training.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import MultiModalBrakingDataset
from data.external_datasets import THURoadSurfaceDataset, MendeleyRoadSurfaceDataset, DAWNWeatherDataset
from utils.preprocessing import DataSynchronizer, PreprocessingConfig
from utils.visualization import SystemAnalyzer


class DataPreprocessor:
    """Preprocess datasets for training."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PreprocessingConfig()
        self.synchronizer = DataSynchronizer(self.config)
        self.mendeley_max_samples = int(os.getenv("MENDELEY_MAX_SAMPLES", "50000"))
        self.dawn_max_samples = int(os.getenv("DAWN_MAX_SAMPLES", "20000"))
    
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
        
        # Preprocess each sample
        processed_samples = []
        
        for i, sample in enumerate(tqdm(dataset.samples, desc="Processing THU samples")):
            try:
                # Process sample
                processed = self.synchronizer.process_sample(sample)
                processed_samples.append(processed)
                
                # Save processed image
                img_output_dir = output_dir / "images"
                img_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save with new filename
                img_path = img_output_dir / f"{i:06d}.pt"
                torch.save(processed['image'], img_path)
                
                # Save metadata
                metadata = {
                    'original_path': sample['image_path'],
                    'surface_type': sample['surface_type'],
                    'surface_id': sample['surface_id'],
                    'mu': sample['mu'],
                    'timestamp': sample.get('timestamp', 0)
                }
                
            except Exception as e:
                print(f"[WARNING] Error processing sample {i}: {e}")
                continue
        
        # Save CAN data (dummy for THU since it doesn't have CAN data)
        can_data = pd.DataFrame({
            'timestamp': [s.get('timestamp', i) for i, s in enumerate(dataset.samples)],
            'surface_type': [s['surface_type'] for s in dataset.samples],
            'mu': [s['mu'] for s in dataset.samples]
        })
        can_data.to_csv(output_dir / "can_data.csv", index=False)
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            import json
            json.dump({
                'dataset': 'thu_road_surface',
                'total_samples': len(processed_samples),
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"[OK] Preprocessed THU dataset saved to: {output_dir}")
        print(f"   Total samples: {len(processed_samples)}")
        
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

            # Keep shape compatible with temporal/fusion branches.
            dummy_can = torch.zeros(self.config.seq_length, len(self.config.can_signals), dtype=torch.float32)
            torch.save(dummy_can, can_output_dir / f"{i:06d}.pt")

            targets.append(float(sample['mu']))
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

            dummy_can = torch.zeros(self.config.seq_length, len(self.config.can_signals), dtype=torch.float32)
            torch.save(dummy_can, can_output_dir / f"{i:06d}.pt")

            targets.append(float(sample['mu']))
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
        
        # Process data
        processed_data = self.synchronizer.create_dataset(dataset.synchronized_data)
        
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
                'seq_length': self.config.seq_length,
                'processed_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"[OK]Preprocessed custom dataset saved to: {output_dir}")
        print(f"   Total samples: {len(processed_data['images'])}")
        
        return output_dir
    
    def preprocess_all(self):
        """Preprocess all available datasets."""
        processed_dirs = []
        
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
        
        # Custom dataset
        custom_dir = "data/train"
        if os.path.exists(custom_dir):
            processed_dirs.append(self.preprocess_custom_dataset(custom_dir))
        else:
            print(f"[INFO] Custom dataset not found at: {os.path.abspath(custom_dir)}")
        
        return processed_dirs


def main():
    """Main preprocessing function."""
    import torch
    import pandas as pd
    
    preprocessor = DataPreprocessor()
    processed_dirs = preprocessor.preprocess_all()
    
    print("\n" + "="*70)
    print("[OK] Data preprocessing complete!")
    print(f"Processed datasets saved to: {preprocessor.output_dir}")
    for d in processed_dirs:
        print(f"  - {d}")
    print("="*70)


if __name__ == "__main__":
    main()