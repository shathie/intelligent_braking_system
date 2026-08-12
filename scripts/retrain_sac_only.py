"""
Retrain only the SAC controller using existing trained upstream models.

Usage:
  python scripts/retrain_sac_only.py --models-dir output/models/20260730_202001
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import ModelTrainer
from models.vit import RoadSurfaceViT
from models.temporal_network import TemporalNetwork
from models.fusion_network import CrossModalAttentionFusion
from models.pinn import PhysicsInformedNetwork


def _resolve_models_dir(models_dir_arg: str) -> Path:
    candidate = Path(models_dir_arg)
    if candidate.exists():
        return candidate

    root = Path("output/models")
    runs = [d for d in root.iterdir() if d.is_dir() and d.name != "latest"] if root.exists() else []
    if not runs:
        raise FileNotFoundError("No model run directories found under output/models")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _load_upstream_models(models_dir: Path, device: torch.device):
    vit = RoadSurfaceViT(num_classes=27, pretrained=False).to(device)
    vit.load_state_dict(torch.load(models_dir / "vit_final.pth", map_location=device))
    vit.eval()

    temporal = TemporalNetwork(input_dim=17, hidden_dim=256, num_layers=2).to(device)
    temporal.load_state_dict(torch.load(models_dir / "temporal_final.pth", map_location=device))
    temporal.eval()

    fusion = CrossModalAttentionFusion(vit_dim=512, temporal_dim=256, fusion_dim=512).to(device)
    fusion.load_state_dict(torch.load(models_dir / "fusion_final.pth", map_location=device))
    fusion.eval()

    pinn = PhysicsInformedNetwork(input_dim=512, hidden_dims=[256, 128, 64]).to(device)
    pinn.load_state_dict(torch.load(models_dir / "pinn_final.pth", map_location=device))
    pinn.eval()

    return vit, temporal, fusion, pinn


def main():
    parser = argparse.ArgumentParser(description="Retrain only SAC using existing upstream checkpoints")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="output/models/20260730_202001",
        help="Directory containing vit/temporal/fusion/pinn final checkpoints",
    )
    args = parser.parse_args()

    trainer = ModelTrainer()
    control_config = trainer.load_config("control_config")

    models_dir = _resolve_models_dir(args.models_dir)
    print(f"Using upstream checkpoints from: {models_dir}")
    print(f"SAC output run directory: {trainer.models_dir}")

    vit_model, temporal_model, fusion_model, pinn_model = _load_upstream_models(models_dir, trainer.device)

    sac_agent = trainer.train_sac(
        control_config,
        pinn_model,
        fusion_model,
        vit_model,
        temporal_model,
    )

    trainer._save_all_models({"sac": sac_agent})
    trainer._save_training_history()

    print("\nSAC-only retraining complete.")
    print(f"New SAC checkpoint: {trainer.models_dir / 'sac_final.pth'}")


if __name__ == "__main__":
    main()
