# Prerequisites

## Create environment
pip create -n braking_system python=3.9
pip activate braking_system

## Install PyTorch with CUDA
pip install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

## Install other dependencies
pip install timm numpy pandas matplotlib opencv-python scikit-learn tqdm pyyaml

## For MPC (optional)
pip install casadi

## For CAN bus interface (optional)
pip install python-can

## Training the Systems

# Train ViT first
python train.py --train-vit --data-dir data/train

# Train full pipeline
python train.py --train-all --data-dir data/train

# Or use main.py
python main.py --train-all --data-dir data/train --save-models

## Running Simulation

python main.py --simulate --output-dir output

## Evaluating Models

python main.py --evaluate --data-dir data/test --output-dir output



# Complete Workflow (Recommended)

# Step 1: Setup environment (only needed once)
python main.py --step setup

# Step 2: Download datasets (manual download recommended)
python main.py --step download

# Step 3: Analyze datasets
python main.py --step analyze

# Step 4: Preprocess data
python main.py --step preprocess

# Step 5: Train all models
python main.py --step train

# Step 6: Evaluate models
python main.py --step evaluate

# Step 7: Run simulations
python main.py --step simulate

# Step 8: Generate reports
python main.py --step report

# Run everything in sequence (takes several hours)
python main.py --all


# Public Dataset Ingestion (KITTI / BDD100K)

# KITTI raw sequence -> custom synchronized dataset format
python scripts/prepare_kitti_for_pipeline.py --kitti-seq-dir data/external/kitti_raw/2011_09_26_drive_0001_sync --output-dir data/train_kitti_0001 --clear-output

# BDD100K videos -> custom synchronized dataset format
python scripts/prepare_bdd100k_for_pipeline.py --video-dir data/external/bdd100k/videos/train --output-dir data/train_bdd100k --sample-fps 5 --clear-output

# Optional: provide BDD labels JSON for weather/scene-informed surface tags
python scripts/prepare_bdd100k_for_pipeline.py --video-dir data/external/bdd100k/videos/train --labels-json data/external/bdd100k/labels/bdd100k_labels_images_train.json --output-dir data/train_bdd100k --sample-fps 5 --clear-output

# Convert custom dataset into processed tensors for training/evaluation
python scripts/preprocess_data.py --only-custom --custom-dir data/train_bdd100k