"""
Complete training script for all models.
Trains ViT, Temporal Network, Fusion Network, PINN, and SAC/MPC in sequence.
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm


def _get_env_int(name: str, default: int = 0) -> int:
    """Read integer environment variable with safe fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import MultiModalBrakingDataset, get_dataloaders
from data.external_datasets import (
    THURoadSurfaceDataset,
    MendeleyVehicleDataset,
    MendeleyRoadSurfaceDataset,
    DAWNWeatherDataset,
    CombinedDataset,
)
from models.vit import RoadSurfaceViT
from models.temporal_network import TemporalNetwork, TransformerTemporalNetwork
from models.fusion_network import CrossModalAttentionFusion, GatedFusion
from models.pinn import PhysicsInformedNetwork
from models.sac_agent import SACAgent
from models.mpc_controller import MPCController
from utils.preprocessing import DataSynchronizer, PreprocessingConfig
from utils.visualization import TrainingPlotter, PredictionVisualizer
from utils.metrics import ClassificationMetrics, RegressionMetrics


class ModelTrainer:
    """Train all models in sequence."""
    
    def __init__(self, config_dir: str = "configs", output_dir: str = "output"):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.models_root = self.output_dir / "models"
        self.models_dir = self.models_root / self.run_id
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir = self.output_dir / "metrics"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Disabled on purpose: do not overwrite or delete model aliases.
        # latest_dir = self.models_root / "latest"
        # if latest_dir.exists() or latest_dir.is_symlink():
        #     if latest_dir.is_symlink() or latest_dir.is_file():
        #         latest_dir.unlink(missing_ok=True)
        #     else:
        #         shutil.rmtree(latest_dir)
        # latest_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize utilities
        self.plotter = TrainingPlotter(self.plots_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_train_samples = _get_env_int("IBS_MAX_TRAIN_SAMPLES", 0)
        self.max_val_samples = _get_env_int("IBS_MAX_VAL_SAMPLES", 0)
        self.epoch_override = _get_env_int("IBS_EPOCH_OVERRIDE", 0)
        self.min_train_batches = _get_env_int("IBS_MIN_TRAIN_BATCHES", 0)
        
        # Training history
        self.history = {
            'vit': {'train_loss': [], 'val_loss': [], 'val_acc': []},
            'temporal': {'train_loss': [], 'val_loss': [], 'val_mae': []},
            'fusion': {'train_loss': [], 'val_loss': [], 'val_mae': []},
            'pinn': {'train_loss': [], 'val_loss': [], 'physics_loss': [], 'val_mae': []},
            'sac': {'train_loss': [], 'reward': []}
        }

    def _maybe_cap_dataset(self, dataset, cap: int, label: str, min_count: int = 0):
        """Cap dataset size for fast experimentation runs."""
        if cap <= 0 and min_count <= 0:
            return dataset

        target = cap if cap > 0 else len(dataset)
        if min_count > 0:
            target = max(target, min_count)

        if len(dataset) <= target:
            return dataset

        from torch.utils.data import Subset
        print(f"[FAST MODE] Capping {label} to {target} samples (from {len(dataset)}).")
        return Subset(dataset, list(range(target)))
    
    def load_config(self, config_name: str) -> Dict:
        """Load configuration file and normalize numeric values."""
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        training_cfg = config.get('training', {})
        if isinstance(training_cfg, dict):
            for key in ['learning_rate', 'weight_decay', 'epochs', 'batch_size', 'warmup_epochs']:
                if key in training_cfg and isinstance(training_cfg[key], str):
                    try:
                        if key in ['epochs', 'batch_size', 'warmup_epochs']:
                            training_cfg[key] = int(float(training_cfg[key]))
                        else:
                            training_cfg[key] = float(training_cfg[key])
                    except ValueError:
                        pass

        model_cfg = config.get('model', {})
        if isinstance(model_cfg, dict):
            for key in ['num_classes', 'patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio']:
                if key in model_cfg and isinstance(model_cfg[key], str):
                    try:
                        model_cfg[key] = int(float(model_cfg[key]))
                    except ValueError:
                        pass

        # Global override for quick experiments.
        if self.epoch_override > 0 and isinstance(training_cfg, dict):
            training_cfg['epochs'] = self.epoch_override

        return config
    
    def save_config(self, config: Dict, config_name: str) -> None:
        """Save configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    def create_dataloaders(self, dataset_type: str = "combined", 
                          batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Create data loaders for training."""
        from torch.utils.data import DataLoader, random_split

        def collate_mixed(batch):
            images = []
            can_sequences = []
            targets = []
            surface_ids = []

            for item in batch:
                images.append(item['image'])
                if 'can_sequence' in item:
                    can_sequences.append(item['can_sequence'])
                else:
                    can_sequences.append(torch.zeros(50, 17))

                target_val = item.get('target_mu', item.get('mu', 0.5))
                if torch.is_tensor(target_val):
                    targets.append(target_val.float().reshape(-1)[0])
                else:
                    targets.append(torch.tensor(float(target_val), dtype=torch.float32))

                surface_val = item.get('surface_id', 0)
                if torch.is_tensor(surface_val):
                    surface_ids.append(surface_val.long().reshape(-1)[0])
                else:
                    surface_ids.append(torch.tensor(int(surface_val), dtype=torch.long))

            return {
                'image': torch.stack(images),
                'can_sequence': torch.stack(can_sequences),
                'target_mu': torch.stack(targets),
                'surface_id': torch.stack(surface_ids)
            }
        
        if dataset_type == "thu":
            dataset = THURoadSurfaceDataset("data/external/thu_road_surface")
            dataset = self._maybe_cap_dataset(
                dataset,
                self.max_train_samples,
                "THU train",
                min_count=self.min_train_batches * batch_size
            )
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=0, collate_fn=collate_mixed)
            val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate_mixed)
            return train_loader, val_loader
        
        elif dataset_type == "mendeley":
            train_dataset = MendeleyRoadSurfaceDataset("data/external/mendeley_vehicle", split='train-set')
            train_dataset = self._maybe_cap_dataset(
                train_dataset,
                self.max_train_samples,
                "Mendeley train",
                min_count=self.min_train_batches * batch_size
            )
            test_split_path = os.path.join("data/external/mendeley_vehicle", "test-set")
            if os.path.exists(test_split_path):
                val_dataset = MendeleyRoadSurfaceDataset("data/external/mendeley_vehicle", split='test-set')
                val_dataset = self._maybe_cap_dataset(val_dataset, self.max_val_samples, "Mendeley val")
            else:
                val_len = max(1, int(0.1 * len(train_dataset)))
                train_len = max(1, len(train_dataset) - val_len)
                train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=0, collate_fn=collate_mixed)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate_mixed)
            return train_loader, val_loader

        elif dataset_type == "dawn":
            dataset = DAWNWeatherDataset("data/external/dawn")
            dataset = self._maybe_cap_dataset(
                dataset,
                self.max_train_samples,
                "DAWN total",
                min_count=self.min_train_batches * batch_size
            )
            val_len = max(1, int(0.2 * len(dataset)))
            train_len = max(1, len(dataset) - val_len)
            train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=0, collate_fn=collate_mixed)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate_mixed)
            return train_loader, val_loader
        
        else:  # combined
            try:
                datasets = []
                weights = []

                if os.path.exists("data/external/thu_road_surface"):
                    datasets.append(THURoadSurfaceDataset("data/external/thu_road_surface"))
                    weights.append(1.0)
                if os.path.exists(os.path.join("data/external/mendeley_vehicle", "train-set")):
                    datasets.append(MendeleyRoadSurfaceDataset("data/external/mendeley_vehicle", split='train-set'))
                    weights.append(2.0)
                if os.path.exists("data/external/dawn"):
                    datasets.append(DAWNWeatherDataset("data/external/dawn"))
                    weights.append(1.0)

                combined = CombinedDataset(datasets, weights=weights)
                combined = self._maybe_cap_dataset(
                    combined,
                    self.max_train_samples,
                    "combined train",
                    min_count=self.min_train_batches * batch_size
                )
                train_loader = DataLoader(combined, batch_size=batch_size, shuffle=True,
                                         num_workers=0, collate_fn=collate_mixed)
                val_dataset = self._maybe_cap_dataset(combined, self.max_val_samples, "combined val")
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=collate_mixed)
                return train_loader, val_loader
                
            except Exception as e:
                print(f"Could not load combined dataset: {e}")
                print("Falling back to Mendeley dataset only...")
                return self.create_dataloaders("mendeley", batch_size)
    
    def train_vit(self, config: Dict, train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader) -> RoadSurfaceViT:
        """Train Vision Transformer for road surface classification."""
        print("\n" + "="*70)
        print("TRAINING VISION TRANSFORMER (ViT)")
        print("="*70)

        # Skip training if checkpoint already exists
        checkpoint_path = self.models_dir / "vit_final.pth"
        if checkpoint_path.exists():
            print(f"[SKIP] vit_final.pth already exists - loading checkpoint.")
            model = RoadSurfaceViT(
                num_classes=config['model']['num_classes'],
                pretrained=False
            ).to(self.device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            return model

        # Initialize model
        model = RoadSurfaceViT(
            num_classes=config['model']['num_classes'],
            pretrained=config['model'].get('pretrained', True)
        ).to(self.device)
        
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(config['training']['epochs']):
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} (ViT)"):
                images = batch['image'].to(self.device)
                surface_ids = batch.get('surface_id', None)
                
                if surface_ids is not None:
                    surface_ids = surface_ids.to(self.device)
                else:
                    # If no surface_id, use dummy labels for feature extraction
                    surface_ids = torch.randint(0, config['model']['num_classes'], 
                                              (images.size(0),)).to(self.device)
                
                # Forward pass
                outputs = model(images)
                
                # Loss
                loss = criterion(outputs['logits'], surface_ids)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    surface_ids = batch.get('surface_id', None)
                    
                    if surface_ids is not None:
                        surface_ids = surface_ids.to(self.device)
                    else:
                        surface_ids = torch.randint(0, config['model']['num_classes'], 
                                                  (images.size(0),)).to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs['logits'], surface_ids)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs['logits'].data, 1)
                    total += surface_ids.size(0)
                    correct += (predicted == surface_ids).sum().item()
            
            val_acc = 100 * correct / total
            
            # Update history
            self.history['vit']['train_loss'].append(train_loss / len(train_loader))
            self.history['vit']['val_loss'].append(val_loss / len(val_loader))
            self.history['vit']['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.models_dir / "vit_best.pth")
                print(f"  New best model saved (Val Acc: {val_acc:.2f}%)")
            
            scheduler.step()
        
        # Save final model
        torch.save(model.state_dict(), self.models_dir / "vit_final.pth")
        print(f"ViT training complete. Best Val Acc: {best_val_acc:.2f}%")
        
        # Plot training curve
        self.plotter.plot_training_curve(
            {'train_loss': self.history['vit']['train_loss'],
             'val_loss': self.history['vit']['val_loss'],
             'val_acc': self.history['vit']['val_acc']},
            title="ViT Training Curve"
        )
        
        return model
    
    def train_temporal_network(self, config: Dict, 
                               train_loader: torch.utils.data.DataLoader,
                               val_loader: torch.utils.data.DataLoader) -> TemporalNetwork:
        """Train temporal network for CAN bus data."""
        print("\n" + "="*70)
        print("TRAINING TEMPORAL NETWORK")
        print("="*70)

        # Skip training if checkpoint already exists
        checkpoint_path = self.models_dir / "temporal_final.pth"
        if checkpoint_path.exists():
            print(f"[SKIP] temporal_final.pth already exists - loading checkpoint.")
            model = TemporalNetwork(
                input_dim=config['model']['input_dim'],
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model'].get('dropout', 0.2)
            ).to(self.device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            return model

        # Initialize model
        model = TemporalNetwork(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model'].get('dropout', 0.2)
        ).to(self.device)
        
        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.001)
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(config['training']['epochs']):
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} (Temporal)"):
                can_sequences = batch['can_sequence'].to(self.device)
                # For temporal network, we predict the last CAN frame from the sequence
                # This is a simplified approach - in practice, you might predict multiple outputs
                
                # Get input shape: (B, seq_len, features)
                # We want to predict the last frame
                inputs = can_sequences[:, :-1, :]  # All but last frame
                targets = can_sequences[:, -1, :]   # Last frame
                
                # Forward pass
                outputs = model(inputs)
                # If model returns (features, recon), use recon for prediction
                if isinstance(outputs, tuple):
                    _, recon = outputs
                else:
                    recon = outputs

                # Loss
                loss = criterion(recon, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    can_sequences = batch['can_sequence'].to(self.device)
                    inputs = can_sequences[:, :-1, :]
                    targets = can_sequences[:, -1, :]
                    
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        _, recon = outputs
                    else:
                        recon = outputs
                    loss = criterion(recon, targets)
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(recon - targets)).item()
            
            # Update history
            self.history['temporal']['train_loss'].append(train_loss / len(train_loader))
            self.history['temporal']['val_loss'].append(val_loss / len(val_loader))
            self.history['temporal']['val_mae'].append(val_mae / len(val_loader))
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f}, "
                  f"Val Loss: {val_loss/len(val_loader):.6f}, "
                  f"Val MAE: {val_mae/len(val_loader):.6f}")
            
            # Save best model
            if val_loss / len(val_loader) < best_val_loss:
                best_val_loss = val_loss / len(val_loader)
                torch.save(model.state_dict(), self.models_dir / "temporal_best.pth")
                print(f"  New best model saved (Val Loss: {best_val_loss:.6f})")
        
        # Save final model
        torch.save(model.state_dict(), self.models_dir / "temporal_final.pth")
        print(f"Temporal network training complete. Best Val Loss: {best_val_loss:.6f}")
        
        # Plot training curve
        self.plotter.plot_training_curve(
            {'train_loss': self.history['temporal']['train_loss'],
             'val_loss': self.history['temporal']['val_loss'],
             'val_mae': self.history['temporal']['val_mae']},
            title="Temporal Network Training Curve"
        )
        
        return model
    
    def train_fusion_network(self, config: Dict, vit_model: RoadSurfaceViT,
                            temporal_model: TemporalNetwork,
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader) -> CrossModalAttentionFusion:
        """Train fusion network."""
        print("\n" + "="*70)
        print("TRAINING FUSION NETWORK")
        print("="*70)
        
        # Freeze ViT and temporal models
        for param in vit_model.parameters():
            param.requires_grad = False
        for param in temporal_model.parameters():
            param.requires_grad = False
        
        # Initialize fusion model
        model = CrossModalAttentionFusion(
            vit_dim=config['model']['vit_dim'],
            temporal_dim=config['model']['temporal_dim'],
            fusion_dim=config['model']['fusion_dim'],
            num_heads=config['model'].get('num_heads', 8)
        ).to(self.device)
        
        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(config['training']['epochs']):
            model.train()
            vit_model.eval()
            temporal_model.eval()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} (Fusion)"):
                images = batch['image'].to(self.device)
                can_sequences = batch['can_sequence'].to(self.device)
                
                # Get features from ViT
                with torch.no_grad():
                    vit_out = vit_model(images)
                    vit_features = vit_out['features']
                
                # Get features from temporal network
                # Process each sequence in the batch
                temporal_features_list = []
                for i in range(can_sequences.size(0)):
                    seq = can_sequences[i:i+1]  # (1, seq_len, features)
                    with torch.no_grad():
                        temp_out = temporal_model(seq)
                    # temporal_model may return (features, recon)
                    temporal_features = temp_out[0] if isinstance(temp_out, tuple) else temp_out
                    temporal_features_list.append(temporal_features)
                temporal_features = torch.cat(temporal_features_list, dim=0)
                
                # Fusion
                fusion_features = model(vit_features, temporal_features)
                
                # For loss, we'll use the target friction as a proxy
                # In practice, you might have a more sophisticated loss
                targets = batch['target_mu'].to(self.device).unsqueeze(1)
                
                # Simple loss: predict friction from fusion features
                # This is a placeholder - in practice, you'd train PINN separately
                pred_mu = torch.sigmoid(torch.mean(fusion_features, dim=1, keepdim=True))
                loss = criterion(pred_mu, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    can_sequences = batch['can_sequence'].to(self.device)
                    
                    vit_out = vit_model(images)
                    vit_features = vit_out['features']
                    
                    temporal_features_list = []
                    for i in range(can_sequences.size(0)):
                        seq = can_sequences[i:i+1]
                        temp_out = temporal_model(seq)
                        temporal_features = temp_out[0] if isinstance(temp_out, tuple) else temp_out
                        temporal_features_list.append(temporal_features)
                    temporal_features = torch.cat(temporal_features_list, dim=0)
                    
                    fusion_features = model(vit_features, temporal_features)
                    targets = batch['target_mu'].to(self.device).unsqueeze(1)
                    pred_mu = torch.sigmoid(torch.mean(fusion_features, dim=1, keepdim=True))
                    loss = criterion(pred_mu, targets)
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(pred_mu - targets)).item()
            
            # Update history
            self.history['fusion']['train_loss'].append(train_loss / len(train_loader))
            self.history['fusion']['val_loss'].append(val_loss / len(val_loader))
            self.history['fusion']['val_mae'].append(val_mae / len(val_loader))
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f}, "
                    f"Val Loss: {val_loss/len(val_loader):.6f}, "
                    f"Val MAE: {val_mae/len(val_loader):.6f}")
            
            # Save best model
            if val_loss / len(val_loader) < best_val_loss:
                best_val_loss = val_loss / len(val_loader)
                torch.save(model.state_dict(), self.models_dir / "fusion_best.pth")
                print(f"  New best model saved (Val Loss: {best_val_loss:.6f})")
        
        # Save final model
        torch.save(model.state_dict(), self.models_dir / "fusion_final.pth")
        print(f"Fusion network training complete. Best Val Loss: {best_val_loss:.6f}")
        
        # Plot training curve
        self.plotter.plot_training_curve(
            {'train_loss': self.history['fusion']['train_loss'],
             'val_loss': self.history['fusion']['val_loss'],
             'val_mae': self.history['fusion']['val_mae']},
            title="Fusion Network Training Curve"
        )
        
        return model
    
    def train_pinn(self, config: Dict, fusion_model: CrossModalAttentionFusion,
                  vit_model: RoadSurfaceViT, temporal_model: TemporalNetwork,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader) -> PhysicsInformedNetwork:
        """Train Physics-Informed Neural Network for friction estimation."""
        print("\n" + "="*70)
        print("TRAINING PHYSICS-INFORMED NEURAL NETWORK (PINN)")
        print("="*70)
        
        # Freeze other models
        for param in vit_model.parameters():
            param.requires_grad = False
        for param in temporal_model.parameters():
            param.requires_grad = False
        for param in fusion_model.parameters():
            param.requires_grad = False
        
        # Initialize PINN
        model = PhysicsInformedNetwork(
            input_dim=config['model']['input_dim'],
            hidden_dims=config['model']['hidden_dims'],
            output_dim=config['model']['output_dim']
        ).to(self.device)
        
        # Loss weights
        alpha = config['physics']['weights']['data_loss']
        beta = config['physics']['weights']['physics_loss']
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        total_epochs = int(config['training']['epochs'])
        warmup_epochs = max(1, min(5, total_epochs // 5))

        for epoch in range(total_epochs):
            model.train()
            fusion_model.eval()
            vit_model.eval()
            temporal_model.eval()
            train_loss = 0.0
            train_mse_total = 0.0
            physics_loss_total = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} (PINN)"):
                images = batch['image'].to(self.device)
                can_sequences = batch['can_sequence'].to(self.device)
                targets = batch['target_mu'].to(self.device)
                
                # Forward pass through feature extractors
                with torch.no_grad():
                    vit_out = vit_model(images)
                    vit_features = vit_out['features']
                    
                    temporal_features_list = []
                    for i in range(can_sequences.size(0)):
                        seq = can_sequences[i:i+1]
                        temp_out = temporal_model(seq)
                        temporal_features = temp_out[0] if isinstance(temp_out, tuple) else temp_out
                        temporal_features_list.append(temporal_features)
                    temporal_features = torch.cat(temporal_features_list, dim=0)
                    
                    fusion_features = fusion_model(vit_features, temporal_features)
                
                # Forward pass through PINN
                pred_mu = model(fusion_features)
                
                # Data loss
                mse_loss = torch.nn.MSELoss()(pred_mu.squeeze(), targets)
                
                # Physics loss
                # Extract vehicle state from CAN data
                vehicle_state = {
                    'a_x': can_sequences[:, -1, 0],  # Longitudinal acceleration
                    'omega': can_sequences[:, -1, 2:6],  # Wheel speeds
                    'v_x': can_sequences[:, -1, 8],   # Vehicle speed
                    'F_brake': can_sequences[:, -1, 7],  # Brake pressure (as proxy for force)
                    'F_z': torch.ones(can_sequences.size(0), 4, device=self.device) * 500  # Normal force per wheel
                }
                physics_loss = model.physics_loss(fusion_features.detach(), vehicle_state)
                
                # Warm up physics term to avoid early collapse.
                physics_weight = beta * min(1.0, float(epoch + 1) / float(warmup_epochs))

                # Total loss
                loss = alpha * mse_loss + physics_weight * physics_loss
                if not torch.isfinite(loss):
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_mse_total += mse_loss.item()
                physics_loss_total += physics_loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_mse_total = 0.0
            val_physics_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    can_sequences = batch['can_sequence'].to(self.device)
                    targets = batch['target_mu'].to(self.device)
                    
                    vit_out = vit_model(images)
                    vit_features = vit_out['features']
                    
                    temporal_features_list = []
                    for i in range(can_sequences.size(0)):
                        seq = can_sequences[i:i+1]
                        temp_out = temporal_model(seq)
                        temporal_features = temp_out[0] if isinstance(temp_out, tuple) else temp_out
                        temporal_features_list.append(temporal_features)
                    temporal_features = torch.cat(temporal_features_list, dim=0)
                    
                    fusion_features = fusion_model(vit_features, temporal_features)
                    pred_mu = model(fusion_features)
                    
                    mse_loss = torch.nn.MSELoss()(pred_mu.squeeze(), targets)
                    
                    vehicle_state = {
                        'a_x': can_sequences[:, -1, 0],
                        'omega': can_sequences[:, -1, 2:6],
                        'v_x': can_sequences[:, -1, 8],
                        'F_brake': can_sequences[:, -1, 7],
                        'F_z': torch.ones(can_sequences.size(0), 4, device=self.device) * 500
                    }
                    physics_loss = model.physics_loss(fusion_features, vehicle_state)
                    
                    physics_weight = beta * min(1.0, float(epoch + 1) / float(warmup_epochs))
                    val_loss += (alpha * mse_loss + physics_weight * physics_loss).item()
                    val_mse_total += mse_loss.item()
                    val_physics_loss += physics_loss.item()
                    val_mae += torch.mean(torch.abs(pred_mu.squeeze() - targets)).item()
            
            # Update history
            avg_train_loss = train_loss / len(train_loader)
            avg_train_mse = train_mse_total / len(train_loader)
            avg_physics_loss = physics_loss_total / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mse = val_mse_total / len(val_loader)
            avg_val_physics = val_physics_loss / len(val_loader)
            
            self.history['pinn']['train_loss'].append(avg_train_loss)
            self.history['pinn']['val_loss'].append(avg_val_loss)
            self.history['pinn']['physics_loss'].append(avg_physics_loss)
            self.history['pinn']['val_mae'].append(val_mae / len(val_loader))
            
            print(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} "
                f"(MSE: {avg_train_mse:.6f}, Physics: {avg_physics_loss:.6f}), "
                f"Val Loss: {avg_val_loss:.6f} "
                f"(MSE: {avg_val_mse:.6f}, Physics: {avg_val_physics:.6f}), "
                f"Val MAE: {val_mae/len(val_loader):.6f}"
            )
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "pinn_best.pth")
                print(f"  New best model saved (Val Loss: {best_val_loss:.6f})")
        
        # Save final model
        torch.save(model.state_dict(), self.models_dir / "pinn_final.pth")
        print(f"PINN training complete. Best Val Loss: {best_val_loss:.6f}")
        
        # Plot training curve
        self.plotter.plot_loss_components(
            {'train_loss': self.history['pinn']['train_loss'],
             'val_loss': self.history['pinn']['val_loss'],
             'physics_loss': self.history['pinn']['physics_loss'],
             'val_mae': self.history['pinn']['val_mae']},
            title="PINN Training Loss Components"
        )
        
        return model
    
    def train_sac(self, config: Dict, pinn_model: PhysicsInformedNetwork,
                 fusion_model: CrossModalAttentionFusion,
                 vit_model: RoadSurfaceViT, temporal_model: TemporalNetwork) -> SACAgent:
        """Train Soft Actor-Critic agent for braking control."""
        print("\n" + "="*70)
        print("TRAINING SOFT ACTOR-CRITIC (SAC) AGENT")
        print("="*70)
        
        # Freeze all feature extractors
        for param in vit_model.parameters():
            param.requires_grad = False
        for param in temporal_model.parameters():
            param.requires_grad = False
        for param in fusion_model.parameters():
            param.requires_grad = False
        for param in pinn_model.parameters():
            param.requires_grad = False
        
        # Initialize SAC agent
        state_dim = 10  # μ, v_x, ω_FL, ω_FR, ω_RL, ω_RR, a_x, ψ̇, δ, brake_pressure
        action_dim = 1  # Braking force (0-1)
        
        sac_cfg = dict(config.get('sac', {}))
        sac_cfg['device'] = str(self.device)
        agent = SACAgent(state_dim, action_dim, sac_cfg)
        
        # Create environment (simplified for training)
        from scripts.simulate import VehicleSimulator

        training_cfg = config.get('training', {}) if isinstance(config, dict) else {}
        num_epochs = int(training_cfg.get('epochs', 50))
        episodes_per_epoch = int(training_cfg.get('episodes_per_epoch', 10))
        max_steps = int(training_cfg.get('max_steps', 500))

        # Optional runtime overrides to accelerate end-to-end experiments.
        num_epochs = _get_env_int("IBS_SAC_EPOCHS", num_epochs)
        episodes_per_epoch = _get_env_int("IBS_SAC_EPISODES_PER_EPOCH", episodes_per_epoch)
        max_steps = _get_env_int("IBS_SAC_MAX_STEPS", max_steps)
        
        # Training loop
        best_reward = float('-inf')
        
        for epoch in range(num_epochs):
            epoch_reward = 0.0
            num_episodes = 0
            
            for _ in range(episodes_per_epoch):
                # Reset environment
                simulator = VehicleSimulator()
                
                # Set random surface
                surfaces = ['dry', 'wet', 'icy', 'rough']
                surface = np.random.choice(surfaces)
                mu = np.random.choice([0.8, 0.4, 0.1, 0.6])
                simulator.set_surface(surface, mu)
                
                # Get initial state
                obs = simulator.step(0)
                state = np.array([
                    mu,  # Friction coefficient (estimated)
                    obs["v_x"],
                    obs["omega"][0],
                    obs["omega"][1],
                    obs["omega"][2],
                    obs["omega"][3],
                    obs["a_x"],
                    obs["psi_dot"],
                    obs["steering"],
                    obs["F_brake"] if "F_brake" in obs else 0.0
                ])
                
                episode_reward = 0.0
                
                for t in range(max_steps):
                    # Get action
                    action = agent.get_action(state)
                    braking_force = np.clip(action[0], 0, 1)
                    
                    # Apply action
                    obs = simulator.step(braking_force)
                    
                    # Get next state
                    next_state = np.array([
                        mu,  # Using true μ for training (in practice, use estimated)
                        obs["v_x"],
                        obs["omega"][0],
                        obs["omega"][1],
                        obs["omega"][2],
                        obs["omega"][3],
                        obs["a_x"],
                        obs["psi_dot"],
                        obs["steering"],
                        braking_force
                    ])
                    
                    # Calculate reward
                    reward = self._calculate_reward(obs, braking_force, simulator.dt)
                    
                    # Check if done
                    done = obs["v_x"] < 0.1 or t >= max_steps - 1
                    
                    # Store transition
                    agent.store_transition(state, action, reward, next_state, done)
                    
                    # Train agent
                    metrics = agent.train_step()
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                epoch_reward += episode_reward
                num_episodes += 1
            
            # Update history
            avg_reward = epoch_reward / num_episodes if num_episodes > 0 else 0
            self.history['sac']['reward'].append(avg_reward)
            
            print(f"Epoch {epoch+1}: Avg Reward: {avg_reward:.2f}")
            
            # Save best agent
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(self.models_dir / "sac_best.pth")
                print(f"  New best agent saved (Avg Reward: {avg_reward:.2f})")
        
        # Save final agent
        agent.save(self.models_dir / "sac_final.pth")
        print(f"SAC training complete. Best Avg Reward: {best_reward:.2f}")
        
        return agent
    
    def _calculate_reward(self, obs: Dict, braking_force: float, dt: float) -> float:
        """Calculate reward for braking control."""
        # Stopping distance component (negative)
        stopping_distance = -0.5 * obs["v_x"] * dt
        
        # Jerk minimization (negative)
        # We don't have previous acceleration, so skip for now
        jerk_penalty = 0.0
        
        # Wheel slip penalty (negative)
        R_w = 0.3
        wheel_speeds = obs["omega"]
        v_x = obs["v_x"]
        slip_penalty = -0.2 * np.mean(np.abs((R_w * wheel_speeds - v_x) / np.maximum(R_w * wheel_speeds, v_x, 1e-6)))
        
        # Stability reward (positive)
        stability_reward = 0.1 * (1 - np.abs(obs["psi_dot"]))
        
        # Aggressive braking reward on high friction (positive)
        # This encourages the agent to brake hard when friction is high
        # In practice, we'd use estimated μ, not true μ
        aggressive_reward = 0.0  # Can't use true μ in reward
        
        reward = stopping_distance + jerk_penalty + slip_penalty + stability_reward + aggressive_reward
        
        return reward
    
    def train_all(self, dataset_type: str = "combined") -> Dict[str, torch.nn.Module]:
        """Train all models in sequence."""
        print("\n" + "="*70)
        print("STARTING FULL TRAINING PIPELINE")
        print("="*70)
        
        # Create data loaders
        train_loader, val_loader = self.create_dataloaders(dataset_type)
        
        # Load configurations
        vit_config = self.load_config("vit_config")
        temporal_config = self.load_config("temporal_config")
        fusion_config = self.load_config("fusion_config")
        pinn_config = self.load_config("pinn_config")
        control_config = self.load_config("control_config")
        
        # Train models in sequence
        models = {}
        
        # 1. Train ViT
        print("\n[STEP 1] Training ViT...")
        vit_model = self.train_vit(vit_config, train_loader, val_loader)
        models['vit'] = vit_model
        
        # 2. Train Temporal Network
        print("\n[STEP 2] Training Temporal Network...")
        temporal_model = self.train_temporal_network(temporal_config, train_loader, val_loader)
        models['temporal'] = temporal_model
        
        # 3. Train Fusion Network
        print("\n[STEP 3] Training Fusion Network...")
        fusion_model = self.train_fusion_network(
            fusion_config, vit_model, temporal_model, train_loader, val_loader
        )
        models['fusion'] = fusion_model
        
        # 4. Train PINN
        print("\n[STEP 4] Training PINN...")
        pinn_model = self.train_pinn(
            pinn_config, fusion_model, vit_model, temporal_model, train_loader, val_loader
        )
        models['pinn'] = pinn_model
        
        # 5. Train SAC Agent
        print("\n[STEP 5] Training SAC Agent...")
        sac_agent = self.train_sac(
            control_config, pinn_model, fusion_model, vit_model, temporal_model
        )
        models['sac'] = sac_agent
        
        # Save all models
        self._save_all_models(models)
        
        # Save training history
        self._save_training_history()
        
        print("\n" + "="*70)
        print("FULL TRAINING PIPELINE COMPLETE!")
        print("="*70)
        
        return models
    
    def _save_all_models(self, models: Dict[str, torch.nn.Module]) -> None:
        """Save all trained models."""
        for name, model in models.items():
            run_path = self.models_dir / f"{name}_final.pth"
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), run_path)
            elif hasattr(model, 'save'):
                model.save(run_path)
        
        print(f"\nAll models saved to: {self.models_dir}")
    
    def _save_training_history(self) -> None:
        """Save training history to file."""
        history_path = self.output_dir / f"training_history_{self.run_id}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for model_name, metrics in self.history.items():
            serializable_history[model_name] = {}
            for metric_name, values in metrics.items():
                serializable_history[model_name][metric_name] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")


def main():
    """Main training function."""
    trainer = ModelTrainer()
    
    # Train all models
    models = trainer.train_all(dataset_type="combined")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Device: {trainer.device}")
    print(f"Models trained: {list(models.keys())}")
    print(f"Models saved to: {trainer.models_dir}")
    print(f"Training history saved to: {trainer.output_dir / f'training_history_{trainer.run_id}.json'}")
    print("="*70)


if __name__ == "__main__":
    main()