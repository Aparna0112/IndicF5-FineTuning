# RunPod Setup Guide for IndicF5 Malayalam Fine-tuning

## 1. RunPod Instance Setup

### Choose the Right Pod
- **GPU**: RTX 4090, A100, or H100 (recommended: A100 40GB)
- **Template**: PyTorch 2.1+ or Custom PyTorch template
- **Storage**: At least 50GB (dataset + model checkpoints)
- **RAM**: 32GB+ recommended

### Recommended RunPod Template
```bash
# Use PyTorch template or create custom with:
# - Ubuntu 22.04
# - CUDA 12.1+
# - PyTorch 2.1+
# - Python 3.9+
```

## 2. Initial Setup Commands

Once your RunPod instance starts, run these commands:

```bash
# Update system
apt update && apt upgrade -y
apt install -y git wget curl htop nvtop

# Clone your repository
git clone <your-repo-url>
cd IndicF5-FineTuning

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets soundfile librosa numpy tqdm wandb accelerate
```

## 3. Environment Setup Script

Save this as `setup_runpod.py`:

```python
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
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "./data/audio",
        "./data/processed", 
        "./logs",
        "./checkpoints",
        "./outputs"
    ]
    
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
            print(f"✓ GPU Available: {gpu_name} (Count: {gpu_count})")
            print(f"✓ CUDA Version: {torch.version.cuda}")
        else:
            print("✗ No GPU available")
    except ImportError:
        print("✗ PyTorch not installed")

if __name__ == "__main__":
    print("Setting up RunPod environment for IndicF5 Malayalam Fine-tuning...")
    
    install_requirements()
    setup_directories()
    check_gpu()
    
    print("\n✓ Setup complete! You can now run the training script.")
```

## 4. Updated Training Configuration

Update your `config.py` for RunPod:

```python
import torch
import os

class Config:
    # Model settings
    model_name = "ai4bharat/IndicF5-TTS"
    language = "malayalam"
    
    # Training settings - adjusted for RunPod
    batch_size = 8 if torch.cuda.get_device_properties(0).total_memory > 20e9 else 4
    learning_rate = 5e-5
    num_epochs = 15
    warmup_steps = 300
    max_length = 200
    gradient_accumulation_steps = 2
    
    # Audio settings
    sample_rate = 24000
    max_audio_length = 8.0  # seconds
    min_audio_length = 1.0  # seconds
    
    # Paths - RunPod optimized
    data_dir = "./data/processed"
    output_dir = "./checkpoints/indicf5_malayalam"
    log_dir = "./logs"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging and saving
    logging_steps = 25
    save_steps = 250
    eval_steps = 250
    max_checkpoints = 3  # Keep only 3 latest checkpoints to save space
    
    # RunPod specific
    use_wandb = True
    wandb_project = "indicf5-malayalam-runpod"
    num_workers = 4
    pin_memory = True
    
    # Data filtering
    max_duration = 10.0  # Filter out audio longer than 10 seconds
    min_text_length = 5   # Minimum characters in transcription
```

## 5. RunPod Execution Script

Create `run_training.sh`:

```bash
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
```

## 6. Dataset Preparation for RunPod

Update `prepare_dataset.py` for better RunPod performance:

```python
import os
from datasets import load_dataset, Dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
import json

def prepare_malayalam_tts_data():
    """Prepare Malayalam TTS dataset optimized for RunPod"""
    print("Loading Malayalam TTS dataset from HuggingFace...")
    
    try:
        # Load dataset
        dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2", split="train")
        print(f"Loaded {len(dataset)} samples")
        
        # Create directories
        os.makedirs("./data/audio", exist_ok=True)
        os.makedirs("./data/processed", exist_ok=True)
        
        # Filter and process data
        processed_data = []
        skipped = 0
        
        for idx, example in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Extract data
                audio_array = example['audio']['array']
                sample_rate = example['audio']['sampling_rate']
                text = example['transcription'].strip()
                
                # Quality filters
                duration = len(audio_array) / sample_rate
                if duration < 1.0 or duration > 10.0:
                    skipped += 1
                    continue
                
                if len(text) < 5 or len(text) > 200:
                    skipped += 1
                    continue
                
                # Save audio file
                audio_path = f"./data/audio/malayalam_{idx:06d}.wav"
                sf.write(audio_path, audio_array, sample_rate)
                
                processed_data.append({
                    'audio_path': audio_path,
                    'text': text,
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'text_length': len(text)
                })
                
                # Progress update
                if idx % 500 == 0:
                    print(f"Processed {idx} samples, skipped {skipped}")
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                skipped += 1
                continue
        
        print(f"Successfully processed {len(processed_data)} samples, skipped {skipped}")
        
        # Create dataset
        processed_dataset = Dataset.from_list(processed_data)
        
        # Split data
        train_test = processed_dataset.train_test_split(test_size=0.1, seed=42)
        
        # Save processed dataset
        train_test.save_to_disk("./data/processed")
        
        # Save statistics
        stats = {
            'total_samples': len(processed_data),
            'train_samples': len(train_test['train']),
            'test_samples': len(train_test['test']),
            'avg_duration': np.mean([item['duration'] for item in processed_data]),
            'avg_text_length': np.mean([item['text_length'] for item in processed_data])
        }
        
        with open('./data/dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset prepared successfully!")
        print(f"Train samples: {len(train_test['train'])}")
        print(f"Test samples: {len(train_test['test'])}")
        
        return train_test
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise

if __name__ == "__main__":
    dataset = prepare_malayalam_tts_data()
```

## 7. Running on RunPod

### Step-by-step execution:

1. **Start RunPod instance** with GPU
2. **Upload your code** or clone from GitHub
3. **Run setup**:
   ```bash
   chmod +x run_training.sh
   python setup_runpod.py
   ```

4. **Prepare dataset**:
   ```bash
   python prepare_dataset.py
   ```

5. **Start training**:
   ```bash
   ./run_training.sh
   ```

### Monitor Training:
```bash
# In separate terminal/tmux session
watch -n 1 nvidia-smi
htop

# Check logs
tail -f logs/training.log

# If using wandb
# Check your wandb dashboard
```

## 8. Important Notes for RunPod

- **Storage**: Make sure you have enough disk space (50GB+)
- **Networking**: Dataset download might take time depending on connection
- **Checkpointing**: Models are saved to `./checkpoints/` - download important checkpoints
- **Memory**: Monitor GPU memory usage with `nvidia-smi`
- **Costs**: Stop the pod when not training to avoid charges

## 9. Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch_size in config.py
2. **Dataset download fails**: Check internet connection, try resuming
3. **Model loading fails**: Check HuggingFace token and model access
4. **Audio processing errors**: Install additional audio libraries

### Performance Tips:
- Use fp16 training if supported
- Increase num_workers if you have sufficient CPU
- Use gradient accumulation for larger effective batch sizes
- Monitor disk usage regularly

## 10. After Training

```bash
# Download your trained model
scp -r ./checkpoints/ local_machine:/path/to/save/

# Test inference
python inference.py

# Clean up to save space
rm -rf ./data/audio/  # Keep processed dataset only
```

This setup should work efficiently on RunPod for fine-tuning IndicF5 with the Malayalam dataset.
