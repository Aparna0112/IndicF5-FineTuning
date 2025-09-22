#!/bin/bash
# start_training.sh - Training launcher script

echo "Starting IndicF5 Malayalam Fine-tuning..."

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="indicf5-malayalam-runpod"

# Check system info
echo "=== System Information ==="
nvidia-smi
echo "Python version: $(python3 --version)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"

# Run setup if needed
if [ ! -d "./checkpoints/indicf5_malayalam" ]; then
    echo "=== Running Initial Setup ==="
    python3 setup_runpod.py
fi

# Prepare dataset if not exists
if [ ! -d "./data/processed" ]; then
    echo "=== Preparing Dataset ==="
    python3 prepare_dataset.py
else
    echo "Dataset already prepared, skipping..."
fi

# Start training with error handling
echo "=== Starting Training ==="
python3 train.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed or stopped!"
