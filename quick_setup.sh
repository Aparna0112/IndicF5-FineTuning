#!/bin/bash
# quick_setup.sh - One-command setup for RunPod

set -e  # Exit on any error

echo "=============================================="
echo "IndicF5 Malayalam TTS - Quick Setup for RunPod"
echo "=============================================="

# Update system
echo "Updating system packages..."
apt update -qq && apt install -y git wget curl htop tree > /dev/null

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null

# Install requirements
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null
else
    pip install transformers datasets soundfile librosa numpy tqdm wandb accelerate huggingface-hub > /dev/null
fi

# Make scripts executable
chmod +x *.sh

# Run setup
echo "Running project setup..."
python3 setup_runpod.py

echo ""
echo "=============================================="
echo "Setup completed! Next steps:"
echo "=============================================="
echo "1. Prepare dataset: python3 prepare_dataset.py"
echo "2. Start training: ./start_training.sh"
echo "3. Monitor training: ./monitor.sh"
echo "=============================================="
