#!/usr/bin/env python3
# setup_runpod.py - Corrected Python setup script

import os
import subprocess
import sys

def install_requirements():
    """Install all required packages"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "accelerate>=0.23.0",
        "huggingface-hub>=0.17.0"
    ]
    
    print("Installing Python packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "./data/audio",
        "./data/processed", 
        "./logs",
        "./checkpoints/indicf5_malayalam",
        "./outputs"
    ]
    
    print("Creating project directories...")
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU Available: {gpu_name} (Count: {gpu_count})")
            print(f"✓ GPU Memory: {total_memory:.1f} GB")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("✗ No GPU available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def test_imports():
    """Test essential imports"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import soundfile
        import librosa
        print("✓ All essential imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    print("=" * 50)
    print("IndicF5 Malayalam Fine-tuning Setup for RunPod")
    print("=" * 50)
    
    # Check if requirements.txt exists and install from it
    if os.path.exists("requirements.txt"):
        print("Found requirements.txt, installing from file...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                                stdout=subprocess.DEVNULL)
            print("✓ Installed packages from requirements.txt")
        except subprocess.CalledProcessError:
            print("✗ Failed to install from requirements.txt, installing individually...")
            install_requirements()
    else:
        install_requirements()
    
    setup_directories()
    gpu_available = check_gpu()
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    print(f"GPU Available: {'✓' if gpu_available else '✗'}")
    print(f"Imports Working: {'✓' if imports_ok else '✗'}")
    print(f"Directories Created: ✓")
    
    if gpu_available and imports_ok:
        print("\n✓ Setup completed successfully!")
        print("You can now run: python prepare_dataset.py")
    else:
        print("\n✗ Setup completed with issues. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
