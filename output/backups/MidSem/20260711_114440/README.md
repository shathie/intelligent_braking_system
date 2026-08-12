# Prerequisites

## Create conda environment
conda create -n braking_system python=3.9
conda activate braking_system

## Install PyTorch with CUDA
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

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