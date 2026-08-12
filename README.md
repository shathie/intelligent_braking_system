# Intelligent Braking System

Multi-stage research pipeline for friction-aware adaptive braking using:

- Vision Transformer (ViT) for road-surface perception
- Temporal sequence modeling for CAN-like vehicle signals
- Cross-modal fusion
- Physics-informed friction estimation (PINN)
- SAC-based braking control and simulation

## Project Layout

- `configs/`: model and control configs
- `data/`: external, raw, and processed datasets
- `models/`: model definitions
- `scripts/`: preprocessing, training, evaluation, simulation, reporting
- `output/`: metrics, plots, logs, reports, checkpoints

## Environment Setup

### Option A: venv (recommended for this repo)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```powershell
conda create -n braking_system python=3.9 -y
conda activate braking_system
pip install -r requirements.txt
```

Optional dependencies:

```powershell
pip install casadi      # MPC experiments
pip install python-can  # CAN interface utilities
```

## Dataset Preparation

1. Download datasets from the sources below and place them under `data/external/`.

Dataset source URLs:

- THU Road Surface dataset: https://thu-rsxd.com/dxhdiefb/ (use the road-surface release used in your experiments)
- Mendeley vehicle road-surface dataset: https://data.mendeley.com/
- DAWN adverse weather dataset: https://data.mendeley.com/datasets/766ygrbt8y/3
- BDD100K: https://www.bdd100k.com/
- KITTI Raw: https://www.cvlibs.net/datasets/kitti/raw_data.php

Expected local locations:

- `data/external/thu_road_surface/`
- `data/external/mendeley_vehicle/`
- `data/external/dawn/`
- `data/external/bdd100k/`
- `data/external/kitti_raw/`

2. Run dataset analysis:

```powershell
python scripts/analyze_datasets.py
```

3. Run preprocessing:

```powershell
python scripts/preprocess_data.py
```

Public dataset ingestion helpers:

```powershell
python scripts/prepare_kitti_for_pipeline.py --kitti-seq-dir data/external/kitti_raw/2011_09_26_drive_0001_sync --output-dir data/train_kitti_0001 --clear-output
python scripts/prepare_bdd100k_for_pipeline.py --video-dir data/external/bdd100k/videos/train --output-dir data/train_bdd100k --sample-fps 5 --clear-output
```

## Training

Run full staged training:

```powershell
python scripts/train.py
```

Or run orchestrated workflow step-by-step:

```powershell
python main.py --step analyze
python main.py --step preprocess
python main.py --step train
```

## Evaluation and Simulation

```powershell
python scripts/evaluate.py
python scripts/simulate.py
python scripts/report.py
```

Or via orchestrator:

```powershell
python main.py --step evaluate
python main.py --step simulate
python main.py --step report
```

## Full Pipeline

```powershell
python main.py --all
```

## Practical Runtime Notes

- Full-data runs are compute-intensive.
- For faster experiments, sample caps can be controlled through environment variables used by training/evaluation scripts (for example `IBS_MAX_TRAIN_SAMPLES`, `IBS_MAX_VAL_SAMPLES`, `IBS_MAX_EVAL_SAMPLES`).
- Generated artifacts are written to `output/`.