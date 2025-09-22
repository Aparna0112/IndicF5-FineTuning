#!/bin/bash

echo "Starting IndicF5 Malayalam Fine-tuning on RunPod..."

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="indicf5-malayalam-runpod"

# Login to Wandb (optional - enter your API key)
# wandb login

# Check system info
echo "=== System Information ==="
nvidia-smi
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run setup
echo "=== Running Setup ==="
python setup_runpod.py

# Prepare dataset
echo "=== Preparing Dataset ==="
python prepare_dataset.py

# Start training
echo "=== Starting Training ==="
python train.py

echo "Training completed!"
