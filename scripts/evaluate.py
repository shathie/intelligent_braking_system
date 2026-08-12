"""
Model evaluation script.
Evaluates all trained models on test data and generates performance reports.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import MultiModalBrakingDataset, get_dataloaders
from data.external_datasets import (
    THURoadSurfaceDataset,
    MendeleyVehicleDataset,
    MendeleyRoadSurfaceDataset,
    DAWNWeatherDataset,
    BDD100KDataset,
    KITTIRawDataset,
    CombinedDataset,
)
from models.vit import RoadSurfaceViT
from models.temporal_network import TemporalNetwork
from models.fusion_network import CrossModalAttentionFusion
from models.pinn import PhysicsInformedNetwork
from models.sac_agent import SACAgent
from utils.visualization import PredictionVisualizer, SystemAnalyzer
from utils.metrics import ClassificationMetrics, RegressionMetrics, ControlMetrics, SystemMetrics


ROAD_SURFACE_CLASSES_27 = {
    'dry_asphalt_severe', 'dry_asphalt_slight', 'dry_asphalt_smooth',
    'dry_concrete_severe', 'dry_concrete_slight', 'dry_concrete_smooth',
    'dry_gravel', 'dry_mud', 'fresh_snow', 'ice', 'melted_snow',
    'water_asphalt_severe', 'water_asphalt_slight', 'water_asphalt_smooth',
    'water_concrete_severe', 'water_concrete_slight', 'water_concrete_smooth',
    'water_gravel', 'water_mud',
    'wet_asphalt_severe', 'wet_asphalt_slight', 'wet_asphalt_smooth',
    'wet_concrete_severe', 'wet_concrete_slight', 'wet_concrete_smooth',
    'wet_gravel', 'wet_mud'
}


def _get_env_int(name: str, default: int = 0) -> int:
    """Read integer environment variable with safe fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    """Read boolean environment variable with safe fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ModelEvaluator:
    """Evaluate all trained models."""
    
    def __init__(self, models_dir: str = "output/models", 
                 output_dir: str = "output", 
                 device: str = None):
        self.models_dir = Path(models_dir)
        if (self.models_dir / "latest").exists():
            self.models_dir = self.models_dir / "latest"
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        force_cpu = _get_env_bool("IBS_FORCE_CPU", True)
        if device:
            self.device = torch.device(device)
        elif force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_eval_samples = _get_env_int("IBS_MAX_EVAL_SAMPLES", 0)
        self.visualizer = PredictionVisualizer(self.plots_dir)
        self.system_analyzer = SystemAnalyzer(self.plots_dir)
        
        # Load models
        self.models = self._load_models()
        
        # Metrics trackers
        self.metrics = SystemMetrics()
        self.results = {}

    @staticmethod
    def _extract_temporal_features(temporal_output: torch.Tensor) -> torch.Tensor:
        """Normalize temporal model outputs to feature tensor."""
        if isinstance(temporal_output, (tuple, list)):
            return temporal_output[0]
        return temporal_output

    def _maybe_cap_dataset(self, dataset, cap: int, label: str):
        """Cap dataset size for quick evaluation runs."""
        if cap <= 0:
            return dataset

        if len(dataset) <= cap:
            return dataset

        from torch.utils.data import Subset
        rng = np.random.default_rng(_get_env_int("IBS_SUBSET_SEED", 42))
        indices = rng.choice(len(dataset), size=cap, replace=False)
        print(
            f"[FAST MODE] Capping {label} to {cap} samples (from {len(dataset)}), "
            f"random subset seed={_get_env_int('IBS_SUBSET_SEED', 42)}."
        )
        return Subset(dataset, indices.tolist())

    def _find_latest_model_path(self, model_name: str) -> Path:
        """Find the newest checkpoint for a model across run directories."""
        direct_path = self.models_dir / f"{model_name}_final.pth"
        candidates = []
        if direct_path.exists():
            candidates.append(direct_path)
        candidates.extend(self.models_dir.glob(f"**/{model_name}_final.pth"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all trained models."""
        models = {}
        
        # ViT
        vit_path = self._find_latest_model_path("vit")
        if vit_path is not None:
            vit = RoadSurfaceViT(num_classes=27, pretrained=False)
            vit.load_state_dict(torch.load(vit_path, map_location=self.device))
            vit.to(self.device).eval()
            models['vit'] = vit
        
        # Temporal Network
        temporal_path = self._find_latest_model_path("temporal")
        if temporal_path is not None:
            temporal = TemporalNetwork(input_dim=17, hidden_dim=256, num_layers=2)
            temporal.load_state_dict(torch.load(temporal_path, map_location=self.device))
            temporal.to(self.device).eval()
            models['temporal'] = temporal
        
        # Fusion Network
        fusion_path = self._find_latest_model_path("fusion")
        if fusion_path is not None:
            fusion = CrossModalAttentionFusion(vit_dim=512, temporal_dim=256, fusion_dim=512)
            fusion.load_state_dict(torch.load(fusion_path, map_location=self.device))
            fusion.to(self.device).eval()
            models['fusion'] = fusion
        
        # PINN
        pinn_path = self._find_latest_model_path("pinn")
        if pinn_path is not None:
            pinn = PhysicsInformedNetwork(input_dim=512, hidden_dims=[256, 128, 64])
            pinn.load_state_dict(torch.load(pinn_path, map_location=self.device))
            pinn.to(self.device).eval()
            models['pinn'] = pinn
        
        # SAC Agent
        sac_path = self._find_latest_model_path("sac")
        if sac_path is not None:
            sac = SACAgent(state_dim=10, action_dim=1, config={"device": str(self.device)})
            sac.load(sac_path)
            models['sac'] = sac
        
        return models

    @staticmethod
    def _to_python_scalars(metrics_dict: Dict) -> Dict:
        """Convert numpy scalars/arrays to JSON-serializable Python values."""
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics_dict.items()
        }
    
    def create_test_loader(self, dataset_type: str = "combined", batch_size: int = 32):
        """Create test data loader."""
        from torch.utils.data import DataLoader

        def _mendeley_split_has_road_classes(split_name: str) -> bool:
            split_path = os.path.join("data/external/mendeley_vehicle", split_name)
            if not os.path.exists(split_path):
                return False
            class_dirs = [
                d for d in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, d))
            ]
            if not class_dirs:
                return False
            return all(d.lower() in ROAD_SURFACE_CLASSES_27 for d in class_dirs)

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

                # Keep only canonical 27-class road labels for ViT metrics.
                surface_type = str(item.get('surface_type', '')).lower()
                if surface_type in ROAD_SURFACE_CLASSES_27:
                    surface_val = item.get('surface_id', -1)
                    if torch.is_tensor(surface_val):
                        surface_ids.append(surface_val.long().reshape(-1)[0])
                    else:
                        surface_ids.append(torch.tensor(int(surface_val), dtype=torch.long))
                else:
                    surface_ids.append(torch.tensor(-1, dtype=torch.long))

            return {
                'image': torch.stack(images),
                'can_sequence': torch.stack(can_sequences),
                'target_mu': torch.stack(targets),
                'surface_id': torch.stack(surface_ids)
            }

        if dataset_type == "thu":
            dataset = THURoadSurfaceDataset("data/external/thu_road_surface")
            dataset = self._maybe_cap_dataset(dataset, self.max_eval_samples, "THU eval")
            return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_mixed)
        
        elif dataset_type == "mendeley":
            split = 'test-set' if _mendeley_split_has_road_classes('test-set') else 'train-set'
            dataset = MendeleyRoadSurfaceDataset("data/external/mendeley_vehicle", split=split)
            dataset = self._maybe_cap_dataset(dataset, self.max_eval_samples, "Mendeley eval")
            return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_mixed)

        elif dataset_type == "dawn":
            dataset = DAWNWeatherDataset("data/external/dawn")
            dataset = self._maybe_cap_dataset(dataset, self.max_eval_samples, "DAWN eval")
            return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_mixed)
        
        else:  # combined
            try:
                datasets = []
                weights = []
                if os.path.exists("data/external/thu_road_surface"):
                    datasets.append(THURoadSurfaceDataset("data/external/thu_road_surface"))
                    weights.append(1.0)
                if _mendeley_split_has_road_classes('test-set'):
                    datasets.append(MendeleyRoadSurfaceDataset("data/external/mendeley_vehicle", split='test-set'))
                    weights.append(2.0)
                elif os.path.exists(os.path.join("data/external/mendeley_vehicle", "train-set")):
                    datasets.append(MendeleyRoadSurfaceDataset("data/external/mendeley_vehicle", split='train-set'))
                    weights.append(2.0)
                if os.path.exists("data/external/dawn"):
                    datasets.append(DAWNWeatherDataset("data/external/dawn"))
                    weights.append(1.0)
                try:
                    if os.path.exists("data/external/bdd100k"):
                        datasets.append(BDD100KDataset("data/external/bdd100k"))
                        weights.append(1.5)
                except Exception as _e:
                    print(f"[INFO] BDD100K skipped during eval: {_e}")
                try:
                    if os.path.exists("data/external/kitti_raw"):
                        datasets.append(KITTIRawDataset("data/external/kitti_raw"))
                        weights.append(1.0)
                except Exception as _e:
                    print(f"[INFO] KITTI skipped during eval: {_e}")

                combined = CombinedDataset(datasets, weights=weights)
                combined = self._maybe_cap_dataset(combined, self.max_eval_samples, "combined eval")
                return DataLoader(combined, batch_size=batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate_mixed)
            except Exception as e:
                print(f"[WARNING] Could not load combined dataset: {e}")
                return self.create_test_loader("mendeley", batch_size)
    
    def evaluate_vit(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate ViT model."""
        print("\n" + "="*70)
        print("EVALUATING VISION TRANSFORMER (ViT)")
        print("="*70)
        
        if 'vit' not in self.models:
            print("[WARNING] ViT model not found. Skipping evaluation.")
            return {}
        
        vit = self.models['vit']
        vit.eval()
        
        y_true = []
        y_pred = []
        all_features = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                surface_ids = batch.get('surface_id', None)
                
                if surface_ids is not None:
                    surface_ids = surface_ids.to(self.device)
                else:
                    continue
                
                outputs = vit(images)
                
                _, predicted = torch.max(outputs['logits'].data, 1)

                valid_mask = surface_ids >= 0
                if valid_mask.sum().item() == 0:
                    continue
                
                y_true.extend(surface_ids[valid_mask].cpu().numpy())
                y_pred.extend(predicted[valid_mask].cpu().numpy())
                all_features.extend(outputs['features'][valid_mask].cpu().numpy())
        
        # Compute metrics
        class_names = ['dry_asphalt_severe', 'dry_asphalt_slight', 'dry_asphalt_smooth',
                       'dry_concrete_severe', 'dry_concrete_slight', 'dry_concrete_smooth',
                       'dry_gravel', 'dry_mud', 'fresh_snow', 'ice', 'melted_snow',
                       'water_asphalt_severe', 'water_asphalt_slight', 'water_asphalt_smooth',
                       'water_concrete_severe', 'water_concrete_slight', 'water_concrete_smooth',
                       'water_gravel', 'water_mud',
                       'wet_asphalt_severe', 'wet_asphalt_slight', 'wet_asphalt_smooth',
                       'wet_concrete_severe', 'wet_concrete_slight', 'wet_concrete_smooth',
                       'wet_gravel', 'wet_mud']
        metrics = ClassificationMetrics(num_classes=27, class_names=class_names)
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        class_metrics = metrics.compute(y_true_arr, y_pred_arr)
        
        # Guard: skip if no labelled samples found
        if len(y_true_arr) == 0:
            print("[WARNING] No labelled surface samples found. Skipping ViT metrics.")
            return {}

        # Print report
        metrics.print_report(y_true_arr, y_pred_arr)
        
        # Save metrics
        self.results['vit'] = class_metrics
        with open(self.metrics_dir / "vit_metrics.json", 'w') as f:
            json.dump(class_metrics, f, indent=2)
        
        # Plot confusion matrix
        self.system_analyzer.plot_confusion_matrix(y_true_arr, y_pred_arr, class_names, 
                                                   title="ViT Surface Classification")
        
        print(f"[OK] ViT evaluation complete. Metrics saved to: {self.metrics_dir / 'vit_metrics.json'}")
        
        return class_metrics

    def evaluate_temporal(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate temporal network sequence reconstruction quality."""
        print("\n" + "="*70)
        print("EVALUATING TEMPORAL NETWORK")
        print("="*70)

        if 'temporal' not in self.models:
            print("[WARNING] Temporal model not found. Skipping evaluation.")
            return {}

        temporal = self.models['temporal']
        temporal.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in test_loader:
                can_sequences = batch['can_sequence'].to(self.device)
                if can_sequences.size(1) < 2:
                    continue

                inputs = can_sequences[:, :-1, :]
                targets = can_sequences[:, -1, :]

                temporal_out = temporal(inputs)
                if isinstance(temporal_out, (tuple, list)):
                    _, recon = temporal_out
                else:
                    recon = temporal_out

                target_dim = min(targets.size(-1), recon.size(-1))
                y_true.extend(targets[:, :target_dim].reshape(-1).cpu().numpy().tolist())
                y_pred.extend(recon[:, :target_dim].reshape(-1).cpu().numpy().tolist())

        if len(y_true) == 0:
            print("[WARNING] No temporal samples found. Skipping temporal metrics.")
            return {}

        metrics = RegressionMetrics()
        y_true_arr = np.asarray(y_true, dtype=np.float32)
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        temporal_metrics = self._to_python_scalars(metrics.compute(y_true_arr, y_pred_arr))

        metrics.print_report(y_true_arr, y_pred_arr)

        self.results['temporal'] = temporal_metrics
        with open(self.metrics_dir / "temporal_metrics.json", 'w') as f:
            json.dump(temporal_metrics, f, indent=2)

        print(f"[OK] Temporal evaluation complete. Metrics saved to: {self.metrics_dir / 'temporal_metrics.json'}")
        return temporal_metrics

    def evaluate_fusion(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate fusion branch as direct friction proxy regression."""
        print("\n" + "="*70)
        print("EVALUATING FUSION NETWORK")
        print("="*70)

        required_models = ['vit', 'temporal', 'fusion']
        if not all(m in self.models for m in required_models):
            missing = [m for m in required_models if m not in self.models]
            print(f"[WARNING] Missing models for fusion evaluation: {missing}. Skipping.")
            return {}

        vit = self.models['vit']
        temporal = self.models['temporal']
        fusion = self.models['fusion']

        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                can_sequences = batch['can_sequence'].to(self.device)
                targets = batch['target_mu'].to(self.device)

                vit_out = vit(images)
                vit_features = vit_out['features']

                temporal_features_list = []
                for i in range(can_sequences.size(0)):
                    seq = can_sequences[i:i+1]
                    temporal_out = temporal(seq)
                    temporal_features = self._extract_temporal_features(temporal_out)
                    temporal_features_list.append(temporal_features)
                temporal_features = torch.cat(temporal_features_list, dim=0)

                fusion_features = fusion(vit_features, temporal_features)
                pred_mu = torch.sigmoid(torch.mean(fusion_features, dim=1))

                y_true.extend(targets.reshape(-1).cpu().numpy().tolist())
                y_pred.extend(pred_mu.reshape(-1).cpu().numpy().tolist())

        if len(y_true) == 0:
            print("[WARNING] No fusion samples found. Skipping fusion metrics.")
            return {}

        metrics = RegressionMetrics()
        y_true_arr = np.asarray(y_true, dtype=np.float32)
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        fusion_metrics = self._to_python_scalars(metrics.compute(y_true_arr, y_pred_arr))

        metrics.print_report(y_true_arr, y_pred_arr)

        self.results['fusion'] = fusion_metrics
        with open(self.metrics_dir / "fusion_metrics.json", 'w') as f:
            json.dump(fusion_metrics, f, indent=2)

        print(f"[OK] Fusion evaluation complete. Metrics saved to: {self.metrics_dir / 'fusion_metrics.json'}")
        return fusion_metrics
    
    def evaluate_pinn(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate PINN model."""
        print("\n" + "="*70)
        print("EVALUATING PHYSICS-INFORMED NEURAL NETWORK (PINN)")
        print("="*70)
        
        required_models = ['vit', 'temporal', 'fusion', 'pinn']
        if not all(m in self.models for m in required_models):
            missing = [m for m in required_models if m not in self.models]
            print(f"[WARNING] Missing models for PINN evaluation: {missing}. Skipping.")
            return {}
        
        vit = self.models['vit']
        temporal = self.models['temporal']
        fusion = self.models['fusion']
        pinn = self.models['pinn']
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                can_sequences = batch['can_sequence'].to(self.device)
                targets = batch['target_mu'].to(self.device)
                
                # Get features
                vit_out = vit(images)
                vit_features = vit_out['features']
                
                temporal_features_list = []
                for i in range(can_sequences.size(0)):
                    seq = can_sequences[i:i+1]
                    temporal_out = temporal(seq)
                    temporal_features = self._extract_temporal_features(temporal_out)
                    temporal_features_list.append(temporal_features)
                temporal_features = torch.cat(temporal_features_list, dim=0)
                
                fusion_features = fusion(vit_features, temporal_features)
                
                # Predict friction
                pred_mu = pinn(fusion_features)
                
                y_true.extend(targets.reshape(-1).cpu().numpy().tolist())
                y_pred.extend(pred_mu.reshape(-1).cpu().numpy().tolist())

        if len(y_true) == 0:
            print("[WARNING] No PINN samples found. Skipping PINN metrics.")
            return {}

        y_true_arr = np.asarray(y_true, dtype=np.float32)
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)

        # Diagnostic: warn when predictions or targets are constant (would yield MSE≈0)
        pred_std = float(np.std(y_pred_arr))
        true_std = float(np.std(y_true_arr))
        if pred_std < 1e-5:
            print(f"[WARNING] PINN predictions are nearly constant "
                  f"(std={pred_std:.2e}). Model may have collapsed.")
        if true_std < 1e-5:
            print(f"[WARNING] PINN targets are nearly constant "
                  f"(std={true_std:.2e}). Test set may lack label diversity.")
        
        # Compute metrics
        metrics_obj = RegressionMetrics()
        reg_metrics = self._to_python_scalars(metrics_obj.compute(y_true_arr, y_pred_arr))
        
        # Print report
        metrics_obj.print_report(y_true_arr, y_pred_arr)
        
        # Save metrics
        self.results['pinn'] = reg_metrics
        with open(self.metrics_dir / "pinn_metrics.json", 'w') as f:
            json.dump(reg_metrics, f, indent=2)
        
        # Plot predictions
        self.visualizer.plot_friction_prediction(y_true_arr, y_pred_arr, 
                                                  title="PINN Friction Estimation")
        
        print(f"[OK] PINN evaluation complete. Metrics saved to: {self.metrics_dir / 'pinn_metrics.json'}")
        
        return reg_metrics
    
    def evaluate_sac(self, num_episodes: int = 10) -> Dict:
        """Evaluate SAC agent in simulation."""
        print("\n" + "="*70)
        print("EVALUATING SOFT ACTOR-CRITIC (SAC) AGENT")
        print("="*70)
        
        if 'sac' not in self.models:
            print("[WARNING] SAC agent not found. Skipping evaluation.")
            return {}
        
        sac = self.models['sac']
        max_steps = _get_env_int("IBS_SAC_EVAL_MAX_STEPS", 3000)
        
        # Create simulator
        from scripts.simulate import VehicleSimulator
        
        # Test on different surfaces
        surfaces = [
            ('dry', 0.8),
            ('wet', 0.4),
            ('icy', 0.1),
            ('rough', 0.6)
        ]
        
        results = {}
        
        for surface, true_mu in surfaces:
            print(f"\nTesting on {surface} surface (mu = {true_mu})...")
            
            stopping_distances = []
            stopping_times = []
            max_jerks = []
            stop_successes = 0
            timeout_count = 0
            
            for episode in range(num_episodes):
                simulator = VehicleSimulator()
                simulator.set_surface(surface, true_mu)
                
                # Run simulation
                states = []
                actions = []
                rewards = []
                
                obs = simulator.step(0)
                state = np.array([
                    true_mu,  # Using true mu (in practice, use estimated)
                    obs["v_x"],
                    obs["omega"][0],
                    obs["omega"][1],
                    obs["omega"][2],
                    obs["omega"][3],
                    obs["a_x"],
                    obs["psi_dot"],
                    obs["steering"],
                    0.0
                ])
                
                initial_velocity = obs["v_x"]
                start_time = simulator.time
                prev_accel = obs["a_x"]
                
                for t in range(max_steps):
                    action = sac.get_action(state, deterministic=True)
                    braking_force = np.clip(action[0], 0, 1)
                    
                    obs = simulator.step(braking_force)
                    
                    # Calculate jerk
                    current_accel = obs["a_x"]
                    jerk = (current_accel - prev_accel) / simulator.dt if simulator.dt > 0 else 0
                    prev_accel = current_accel
                    
                    next_state = np.array([
                        true_mu,
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
                    
                    states.append(state.copy())
                    actions.append(braking_force)
                    
                    # Simple reward
                    reward = -0.1 * obs["v_x"] * simulator.dt  # Reward for slowing down
                    rewards.append(reward)
                    
                    state = next_state
                    
                    if obs["v_x"] < 0.1:
                        stop_successes += 1
                        break
                else:
                    timeout_count += 1
                
                # Calculate metrics
                stopping_time = simulator.time - start_time
                stopping_distance = np.trapezoid([s[1] for s in states], [t * simulator.dt for t in range(len(states))])
                max_jerk = np.max([abs((states[i][6] - states[i-1][6]) / simulator.dt) if i > 0 and simulator.dt > 0 else 0 
                                  for i in range(1, len(states))])
                
                stopping_distances.append(stopping_distance)
                stopping_times.append(stopping_time)
                max_jerks.append(max_jerk)
            
            results[surface] = {
                'stopping_distance': np.mean(stopping_distances),
                'stopping_distance_std': np.std(stopping_distances),
                'stopping_time': np.mean(stopping_times),
                'stopping_time_std': np.std(stopping_times),
                'max_jerk': np.mean(max_jerks),
                'max_jerk_std': np.std(max_jerks),
                'stop_success_rate': float(stop_successes) / float(max(num_episodes, 1)),
                'timeout_count': int(timeout_count),
                'eval_max_steps': int(max_steps),
                'num_episodes': num_episodes
            }
        
        # Print results
        print("\n" + "-"*70)
        print("SAC AGENT EVALUATION RESULTS")
        print("-"*70)
        for surface, metrics in results.items():
            print(f"\n{surface.upper()}:")
            print(f"  Stopping Distance: {metrics['stopping_distance']:.2f} +/- {metrics['stopping_distance_std']:.2f} m")
            print(f"  Stopping Time: {metrics['stopping_time']:.2f} +/- {metrics['stopping_time_std']:.2f} s")
            print(f"  Max Jerk: {metrics['max_jerk']:.2f} +/- {metrics['max_jerk_std']:.2f} m/s^3")
            print(f"  Stop Success Rate: {metrics['stop_success_rate']*100:.1f}%")
            print(f"  Timeouts: {metrics['timeout_count']} / {metrics['num_episodes']} (max_steps={metrics['eval_max_steps']})")
        
        # Save results
        self.results['sac'] = results
        with open(self.metrics_dir / "sac_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot braking performance using surface names as keys.
        surface_names = [name for name, _ in surfaces]
        times = [np.linspace(0, max(float(results[name]['stopping_time']), 1e-6), 100) for name in surface_names]
        velocities = [np.maximum(20 - 4 * t, 0) for t in times]
        braking_forces = []
        for t, name in zip(times, surface_names):
            stop_t = max(float(results[name]['stopping_time']), 1e-6)
            braking_forces.append(np.clip(0.8 * (1 - t / stop_t), 0, 1))

        self.visualizer.plot_braking_simulation(
            times=times,
            velocities=velocities,
            braking_forces=braking_forces,
            surface_types=surface_names,
            title="SAC Agent Braking Performance"
        )
        
        print(f"\n[OK] SAC evaluation complete. Metrics saved to: {self.metrics_dir / 'sac_metrics.json'}")
        
        return results
    
    def evaluate_all(self, dataset_type: str = "combined") -> Dict:
        """Evaluate all models."""
        print("\n" + "="*70)
        print("STARTING FULL MODEL EVALUATION")
        print("="*70)
        
        # Create test loader
        test_loader = self.create_test_loader(dataset_type)
        
        # Evaluate each model
        self.results = {}
        
        # 1. Evaluate ViT
        print("\n[STEP 1] Evaluating ViT...")
        self.evaluate_vit(test_loader)
        
        # 2. Evaluate Temporal
        print("\n[STEP 2] Evaluating Temporal Network...")
        self.evaluate_temporal(test_loader)

        # 3. Evaluate Fusion
        print("\n[STEP 3] Evaluating Fusion Network...")
        self.evaluate_fusion(test_loader)

        # 4. Evaluate PINN
        print("\n[STEP 4] Evaluating PINN...")
        self.evaluate_pinn(test_loader)
        
        # 5. Evaluate SAC
        print("\n[STEP 5] Evaluating SAC Agent...")
        self.evaluate_sac(num_episodes=10)
        
        # Generate full report
        self._generate_evaluation_report()
        
        print("\n" + "="*70)
        print("[OK] FULL MODEL EVALUATION COMPLETE!")
        print("="*70)
        
        return self.results
    
    def _generate_evaluation_report(self) -> None:
        """Generate HTML evaluation report."""
        report_path = self.output_dir / "evaluation_report.html"
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        overall_performance = self._get_overall_performance()
        vit_results = self._format_vit_results()
        temporal_results = self._format_temporal_results()
        fusion_results = self._format_fusion_results()
        pinn_results = self._format_pinn_results()
        sac_results = self._format_sac_results()
        recommendations = self._generate_evaluation_recommendations()
        
        with open(report_path, 'w') as f:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                    h3 { color: #777; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .section { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    .highlight { background-color: #e7f3fe; padding: 2px 5px; border-radius: 3px; }
                    img { max-width: 100%; height: auto; margin: 10px 0; }
                </style>
            </head>
            <body>
                <h1>Model Evaluation Report</h1>
                <p>Generated on: __GENERATED_AT__</p>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <p><strong>Device:</strong> __DEVICE__</p>
                    <p><strong>Evaluation Date:</strong> __GENERATED_AT__</p>
                    <p><strong>Overall Performance:</strong> <span class="highlight">__OVERALL__</span></p>
                </div>
                
                <div class="section">
                    <h2>Vision Transformer (ViT) Performance</h2>
                    __VIT_RESULTS__
                </div>

                <div class="section">
                    <h2>Temporal Network Performance</h2>
                    __TEMPORAL_RESULTS__
                </div>

                <div class="section">
                    <h2>Fusion Network Performance</h2>
                    __FUSION_RESULTS__
                </div>
                
                <div class="section">
                    <h2>Physics-Informed Neural Network (PINN) Performance</h2>
                    __PINN_RESULTS__
                </div>
                
                <div class="section">
                    <h2>Soft Actor-Critic (SAC) Agent Performance</h2>
                    __SAC_RESULTS__
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <h3>Classification Results</h3>
                    <img src="plots/confusion_matrix.png" alt="Confusion Matrix">
                    <h3>Friction Estimation</h3>
                    <img src="plots/friction_prediction.png" alt="Friction Prediction">
                    <h3>Braking Performance</h3>
                    <img src="plots/braking_simulation.png" alt="Braking Simulation">
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <p>__RECOMMENDATIONS__</p>
                </div>
            </body>
            </html>
            """
            html = html.replace("__GENERATED_AT__", generated_at)
            html = html.replace("__DEVICE__", str(self.device))
            html = html.replace("__OVERALL__", str(overall_performance))
            html = html.replace("__VIT_RESULTS__", str(vit_results))
            html = html.replace("__TEMPORAL_RESULTS__", str(temporal_results))
            html = html.replace("__FUSION_RESULTS__", str(fusion_results))
            html = html.replace("__PINN_RESULTS__", str(pinn_results))
            html = html.replace("__SAC_RESULTS__", str(sac_results))
            html = html.replace("__RECOMMENDATIONS__", str(recommendations))
            f.write(html)
        
        print(f"[OK] Evaluation report generated: {report_path}")
    
    def _get_overall_performance(self) -> str:
        """Get overall performance summary."""
        summaries = []
        
        if 'vit' in self.results:
            vit_acc = self.results['vit'].get('classification_accuracy', self.results['vit'].get('accuracy', None))
            if vit_acc is not None:
                summaries.append(f"ViT Accuracy: {float(vit_acc)*100:.1f}%")

        if 'temporal' in self.results:
            temporal_mae = self.results['temporal'].get('regression_mae', self.results['temporal'].get('mae', None))
            if temporal_mae is not None:
                summaries.append(f"Temporal MAE: {float(temporal_mae):.4f}")

        if 'fusion' in self.results:
            fusion_mae = self.results['fusion'].get('regression_mae', self.results['fusion'].get('mae', None))
            if fusion_mae is not None:
                summaries.append(f"Fusion MAE: {float(fusion_mae):.4f}")
        
        if 'pinn' in self.results:
            pinn_mse = self.results['pinn'].get('regression_mse', self.results['pinn'].get('mse', None))
            if pinn_mse is not None:
                summaries.append(f"PINN MSE: {float(pinn_mse):.4f}")
        
        if 'sac' in self.results:
            avg_distance = np.mean([v['stopping_distance'] for v in self.results['sac'].values()])
            summaries.append(f"Avg Stopping Distance: {avg_distance:.2f}m")
        
        return ", ".join(summaries) if summaries else "No results available"
    
    def _format_vit_results(self) -> str:
        """Format ViT results for HTML report."""
        if 'vit' not in self.results:
            return "<p>ViT evaluation not available.</p>"
        
        metrics = self.results['vit']
        accuracy = metrics.get('classification_accuracy', metrics.get('accuracy', 0.0))

        # Build class list directly from available metric keys so the table stays
        # aligned with the configured classifier taxonomy.
        class_names = sorted(
            suffix
            for key in metrics.keys()
            if key.startswith('precision_')
            for suffix in [key.replace('precision_', '')]
            if suffix not in {'macro', 'weighted'}
        )
        
        html = f"""
        <p><strong>Classification Accuracy:</strong> {accuracy*100:.2f}%</p>
        <h3>Detailed Metrics:</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Precision (Macro)</td><td>{metrics.get('precision_macro', 0):.4f}</td></tr>
            <tr><td>Recall (Macro)</td><td>{metrics.get('recall_macro', 0):.4f}</td></tr>
            <tr><td>F1 Score (Macro)</td><td>{metrics.get('f1_macro', 0):.4f}</td></tr>
            <tr><td>Precision (Weighted)</td><td>{metrics.get('precision_weighted', 0):.4f}</td></tr>
            <tr><td>Recall (Weighted)</td><td>{metrics.get('recall_weighted', 0):.4f}</td></tr>
            <tr><td>F1 Score (Weighted)</td><td>{metrics.get('f1_weighted', 0):.4f}</td></tr>
        </table>
        <h3>Per-Class Performance:</h3>
        <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th></tr>
        """

        for class_name in class_names:
            precision = metrics.get(f'precision_{class_name}', 0)
            recall = metrics.get(f'recall_{class_name}', 0)
            html += f"<tr><td>{class_name}</td><td>{precision:.4f}</td><td>{recall:.4f}</td></tr>"
        
        html += "</table>"
        
        return html
    
    def _format_pinn_results(self) -> str:
        """Format PINN results for HTML report."""
        if 'pinn' not in self.results:
            return "<p>PINN evaluation not available.</p>"
        
        metrics = self.results['pinn']
        mse = metrics.get('regression_mse', metrics.get('mse', 0.0))
        r2 = metrics.get('regression_r2', metrics.get('r2', 0.0))
        rmse = metrics.get('regression_rmse', metrics.get('rmse', 0.0))
        mae = metrics.get('regression_mae', metrics.get('mae', 0.0))
        max_error = metrics.get('regression_max_error', metrics.get('max_error', 0.0))
        median_error = metrics.get('regression_median_error', metrics.get('median_error', 0.0))
        std_error = metrics.get('regression_std_error', metrics.get('std_error', 0.0))
        mse_low = metrics.get('regression_mse_low_friction', metrics.get('mse_low_friction', None))
        mae_low = metrics.get('regression_mae_low_friction', metrics.get('mae_low_friction', None))
        mse_high = metrics.get('regression_mse_high_friction', metrics.get('mse_high_friction', None))
        mae_high = metrics.get('regression_mae_high_friction', metrics.get('mae_high_friction', None))
        
        html = f"""
        <p><strong>Mean Squared Error:</strong> {mse:.6f}</p>
        <p><strong>R^2 Score:</strong> {r2:.6f}</p>
        <h3>Detailed Metrics:</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>RMSE</td><td>{rmse:.6f}</td></tr>
            <tr><td>MAE</td><td>{mae:.6f}</td></tr>
            <tr><td>Max Error</td><td>{max_error:.6f}</td></tr>
            <tr><td>Median Error</td><td>{median_error:.6f}</td></tr>
            <tr><td>Std Error</td><td>{std_error:.6f}</td></tr>
        """

        if mse_low is not None and mae_low is not None:
            html += f"""
            <tr><td>MSE (Low Friction)</td><td>{mse_low:.6f}</td></tr>
            <tr><td>MAE (Low Friction)</td><td>{mae_low:.6f}</td></tr>
            """

        if mse_high is not None and mae_high is not None:
            html += f"""
            <tr><td>MSE (High Friction)</td><td>{mse_high:.6f}</td></tr>
            <tr><td>MAE (High Friction)</td><td>{mae_high:.6f}</td></tr>
            """
        
        html += "</table>"
        
        return html

    def _format_temporal_results(self) -> str:
        """Format temporal results for HTML report."""
        if 'temporal' not in self.results:
            return "<p>Temporal evaluation not available.</p>"

        metrics = self.results['temporal']
        rmse = metrics.get('regression_rmse', metrics.get('rmse', 0))
        mae = metrics.get('regression_mae', metrics.get('mae', 0))
        r2 = metrics.get('regression_r2', metrics.get('r2', 0))

        return f"""
        <p><strong>Sequence Reconstruction RMSE:</strong> {rmse:.6f}</p>
        <p><strong>Sequence Reconstruction MAE:</strong> {mae:.6f}</p>
        <p><strong>R^2 Score:</strong> {r2:.6f}</p>
        """

    def _format_fusion_results(self) -> str:
        """Format fusion results for HTML report."""
        if 'fusion' not in self.results:
            return "<p>Fusion evaluation not available.</p>"

        metrics = self.results['fusion']
        rmse = metrics.get('regression_rmse', metrics.get('rmse', 0))
        mae = metrics.get('regression_mae', metrics.get('mae', 0))
        r2 = metrics.get('regression_r2', metrics.get('r2', 0))

        return f"""
        <p><strong>Friction Proxy RMSE:</strong> {rmse:.6f}</p>
        <p><strong>Friction Proxy MAE:</strong> {mae:.6f}</p>
        <p><strong>R^2 Score:</strong> {r2:.6f}</p>
        """
    
    def _format_sac_results(self) -> str:
        """Format SAC results for HTML report."""
        if 'sac' not in self.results:
            return "<p>SAC evaluation not available.</p>"
        
        results = self.results['sac']
        
        html = """
        <table>
            <tr><th>Surface</th><th>Stopping Distance (m)</th><th>Stopping Time (s)</th><th>Max Jerk (m/s^3)</th></tr>
        """
        
        for surface, metrics in results.items():
            html += f"""
            <tr>
                <td>{surface}</td>
                <td>{metrics['stopping_distance']:.2f} +/- {metrics['stopping_distance_std']:.2f}</td>
                <td>{metrics['stopping_time']:.2f} +/- {metrics['stopping_time_std']:.2f}</td>
                <td>{metrics['max_jerk']:.2f} +/- {metrics['max_jerk_std']:.2f}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
    
    def _generate_evaluation_recommendations(self) -> str:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Check ViT performance
        if 'vit' in self.results:
            acc = self.results['vit'].get('classification_accuracy',
                                          self.results['vit'].get('accuracy', 0))
            if acc < 0.9:
                recommendations.append(
                    "[WARNING] ViT accuracy is below 90%. Consider: "
                    "(1) More training data, (2) Larger model, (3) Better augmentation."
                )
        
        # Check PINN performance
        if 'pinn' in self.results:
            mse = self.results['pinn'].get('regression_mse',
                                           self.results['pinn'].get('mse', 1.0))
            if mse > 0.05:
                recommendations.append(
                    f"[WARNING] PINN MSE is {mse:.4f} (target: <0.05). Consider: "
                    "(1) More physics constraints, (2) Better feature fusion, "
                    "(3) Larger training dataset."
                )
            
            low_friction_key = (
                'regression_mse_low_friction'
                if 'regression_mse_low_friction' in self.results['pinn']
                else 'mse_low_friction'
            )
            if low_friction_key in self.results['pinn']:
                low_mse = self.results['pinn'][low_friction_key]
                if low_mse > 0.03:
                    recommendations.append(
                        "[WARNING] Low friction estimation has high error. "
                        "This is critical for safety. Consider collecting more low-mu data."
                    )
        
        # Check SAC performance
        if 'sac' in self.results:
            for surface, metrics in self.results['sac'].items():
                if metrics['stopping_distance'] > 40:  # From 20 m/s
                    recommendations.append(
                        f"[WARNING] Stopping distance on {surface} surface is >40m. "
                        "Consider tuning the reward function or improving friction estimation."
                    )
                if metrics['max_jerk'] > 5:
                    recommendations.append(
                        f"[WARNING] Max jerk on {surface} surface is >5 m/s^3. "
                        "Consider adding jerk penalty to the reward function."
                    )
        
        if not recommendations:
            recommendations.append(
                "[OK] All models meet performance targets! "
                "Consider deploying to hardware for real-world testing."
            )
        
        return '<br>'.join(recommendations)


def main():
    """Main evaluation function."""
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all(dataset_type="combined")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Device: {evaluator.device}")
    print(f"Results saved to: {evaluator.output_dir}")
    print(f"Evaluation report: {evaluator.output_dir / 'evaluation_report.html'}")
    print("="*70)


if __name__ == "__main__":
    main()