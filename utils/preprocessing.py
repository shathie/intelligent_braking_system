"""
Preprocessing utilities for multi-modal braking system data.
Handles image normalization, CAN bus signal processing, and synchronization.
"""

import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class RoadSurface(Enum):
    """Road surface classification labels."""
    DRY = 0
    WET = 1
    ICY = 2
    ROUGH = 3
    SLIPPERY = 4
    GRAVEL = 5


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    can_signals: List[str] = field(default_factory=lambda: [
        "timestamp", "a_x", "a_y", "a_z",  # IMU
        "omega_FL", "omega_FR", "omega_RL", "omega_RR",  # Wheel speeds
        "steering_angle", "brake_pressure", "v_x", "yaw_rate",  # Vehicle state
        "tire_temp_FL", "tire_temp_FR", "tire_temp_RL", "tire_temp_RR",  # Tire temps
        "mu"  # Friction coefficient (if available)
    ])
    seq_length: int = 50
    sample_rate: int = 100  # Hz


class ImagePreprocessor:
    """Handles preprocessing of underbody camera images."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """Build image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
        
        Returns:
            Preprocessed image tensor (3, H, W)
        """
        if isinstance(image, str):
            # Load from file path; if missing, create a black image
            if not os.path.exists(image):
                h, w = self.config.image_size
                image = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                image = cv2.imread(image)
                if image is None:
                    h, w = self.config.image_size
                    image = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image for torchvision transforms
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Apply transformations
        image_tensor = self.transform(image)
        return image_tensor
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of images."""
        return torch.stack([self.preprocess(img) for img in images])
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Augmentations:
        - Random horizontal flip
        - Random rotation (±5 degrees)
        - Random brightness/contrast
        - Random Gaussian blur
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation
        angle = np.random.uniform(-5, 5)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-20, 20)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random Gaussian blur
        if np.random.rand() > 0.7:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image


class CANSignalPreprocessor:
    """Handles preprocessing of CAN bus signals."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.signal_indices = {signal: idx for idx, signal in enumerate(self.config.can_signals)}
    
    def preprocess_signal(self, signal: np.ndarray, signal_name: str) -> np.ndarray:
        """
        Preprocess a single CAN signal.
        
        Args:
            signal: Raw signal values
            signal_name: Name of the signal
        
        Returns:
            Preprocessed signal
        """
        # Normalize based on signal type
        if signal_name in ["a_x", "a_y", "a_z"]:
            # Acceleration: normalize to [-10, 10] m/s²
            signal = np.clip(signal, -10, 10)
            signal = (signal + 10) / 20  # Scale to [0, 1]
        
        elif signal_name.startswith("omega_"):
            # Wheel speed: normalize to [0, 100] rad/s
            signal = np.clip(signal, 0, 100)
            signal = signal / 100
        
        elif signal_name == "steering_angle":
            # Steering: normalize to [-π, π]
            signal = np.clip(signal, -np.pi, np.pi)
            signal = (signal + np.pi) / (2 * np.pi)
        
        elif signal_name == "brake_pressure":
            # Brake pressure: normalize to [0, 100] bar
            signal = np.clip(signal, 0, 100)
            signal = signal / 100
        
        elif signal_name == "v_x":
            # Vehicle speed: normalize to [0, 50] m/s (~180 km/h)
            signal = np.clip(signal, 0, 50)
            signal = signal / 50
        
        elif signal_name == "yaw_rate":
            # Yaw rate: normalize to [-5, 5] rad/s
            signal = np.clip(signal, -5, 5)
            signal = (signal + 5) / 10
        
        elif signal_name.startswith("tire_temp_"):
            # Tire temperature: normalize to [0, 150] °C
            signal = np.clip(signal, 0, 150)
            signal = signal / 150
        
        elif signal_name == "mu":
            # Friction coefficient: already in [0, 1]
            signal = np.clip(signal, 0, 1)
        
        else:
            # Default: min-max normalization
            if np.max(signal) > np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        return signal.astype(np.float32)
    
    def preprocess_frame(self, frame: Dict[str, float]) -> np.ndarray:
        """
        Preprocess a single CAN frame (dict of signals).
        
        Args:
            frame: Dictionary of signal_name -> value
        
        Returns:
            Preprocessed feature vector
        """
        features = []
        for signal in self.config.can_signals:
            if signal in frame:
                processed = self.preprocess_signal(np.array([frame[signal]]), signal)[0]
                features.append(processed)
            else:
                # Missing signal: use mean or zero
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def create_sequences(self, can_data: np.ndarray, seq_length: int = None) -> np.ndarray:
        """
        Create sequences of CAN data for temporal network.
        
        Args:
            can_data: Array of preprocessed CAN frames (N, num_signals)
            seq_length: Length of each sequence
        
        Returns:
            Array of sequences (N-seq_length+1, seq_length, num_signals)
        """
        seq_length = seq_length or self.config.seq_length
        num_samples = can_data.shape[0]
        num_sequences = num_samples - seq_length + 1
        
        sequences = np.zeros((num_sequences, seq_length, can_data.shape[1]), dtype=np.float32)
        
        for i in range(num_sequences):
            sequences[i] = can_data[i:i+seq_length]
        
        return sequences
    
    def synchronize_data(self, images: List[str], can_data: pd.DataFrame, 
                        time_tolerance: float = 0.01) -> List[Dict]:
        """
        Synchronize images and CAN data based on timestamps.
        
        Args:
            images: List of image file paths (with timestamps in filenames)
            can_data: DataFrame with CAN data and timestamps
            time_tolerance: Maximum allowed time difference for synchronization
        
        Returns:
            List of synchronized data samples
        """
        synchronized = []
        
        for img_path in images:
            # Extract timestamp from image filename
            # Assumes format: timestamp_camera_id.jpg
            try:
                timestamp = float(img_path.split("_")[0])
            except:
                continue
            
            # Find closest CAN data
            time_diff = np.abs(can_data["timestamp"].values - timestamp)
            closest_idx = np.argmin(time_diff)
            
            if time_diff[closest_idx] <= time_tolerance:
                can_row = can_data.iloc[closest_idx]
                synchronized.append({
                    "image_path": img_path,
                    "timestamp": timestamp,
                    "can_data": can_row.to_dict()
                })
        
        return synchronized


class DataSynchronizer:
    """Handles synchronization of multi-modal data streams."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.image_preprocessor = ImagePreprocessor(config)
        self.can_preprocessor = CANSignalPreprocessor(config)
    
    def process_sample(self, sample: Dict) -> Dict:
        """
        Process a single synchronized sample.
        
        Args:
            sample: Dictionary with image_path and can_data
        
        Returns:
            Processed sample with image tensor and CAN sequence
        """
        # Preprocess image
        image_tensor = self.image_preprocessor.preprocess(sample["image_path"])
        
        # Preprocess CAN data
        can_frame = self.can_preprocessor.preprocess_frame(sample["can_data"])
        
        # Get target friction if available
        target_mu = sample["can_data"].get("mu", None)
        if target_mu is not None:
            target_mu = float(target_mu)
        
        return {
            "image": image_tensor,
            "can_frame": can_frame,
            "target_mu": target_mu,
            "timestamp": sample["timestamp"]
        }
    
    def create_dataset(self, synchronized_data: List[Dict], seq_length: int = None) -> Dict:
        """
        Create full dataset from synchronized data.
        
        Args:
            synchronized_data: List of synchronized samples
            seq_length: Length of CAN sequences
        
        Returns:
            Dictionary with images, CAN sequences, and targets
        """
        seq_length = seq_length or self.config.seq_length
        
        images = []
        can_sequences = []
        targets = []
        timestamps = []
        
        # First, collect all CAN frames in order
        all_can_frames = []
        for sample in synchronized_data:
            can_frame = self.can_preprocessor.preprocess_frame(sample["can_data"])
            all_can_frames.append(can_frame)
        
        all_can_frames = np.array(all_can_frames)
        
        # Create sequences
        can_sequences_array = self.can_preprocessor.create_sequences(
            all_can_frames, seq_length
        )
        
        # Match images to sequences
        for i, sample in enumerate(synchronized_data):
            if i >= seq_length - 1:
                # Get the image for this sample
                image_tensor = self.image_preprocessor.preprocess(sample["image_path"])
                images.append(image_tensor)
                
                # Get the corresponding CAN sequence
                # Sequence ends at this sample
                seq_idx = i - (seq_length - 1)
                can_sequences.append(can_sequences_array[seq_idx])
                
                # Get target
                target_mu = sample["can_data"].get("mu", None)
                if target_mu is not None:
                    targets.append(float(target_mu))
                else:
                    targets.append(None)
                
                timestamps.append(sample["timestamp"])

        if not can_sequences:
            raise ValueError(
                f"No CAN sequences generated. samples={len(synchronized_data)}, seq_length={seq_length}."
            )

        can_sequences_np = np.stack(can_sequences, axis=0).astype(np.float32, copy=False)
        
        return {
            "images": torch.stack(images),
            "can_sequences": torch.from_numpy(can_sequences_np),
            "targets": torch.tensor([t if t is not None else 0.0 for t in targets], dtype=torch.float32),
            "timestamps": timestamps
        }
