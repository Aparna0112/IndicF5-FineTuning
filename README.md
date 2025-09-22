# IndicF5 Malayalam Fine-tuning

Fine-tune the IndicF5 Text-to-Speech model for Malayalam language using the MALAYALAM_STT_COMBINED_1_2 dataset on RunPod.

## 🚀 Quick Start on RunPod

### 1. Launch RunPod Instance
- **GPU**: RTX 4090, A100, or H100 (recommended: A100 40GB)
- **Template**: PyTorch 2.1+ template
- **Storage**: At least 50GB

### 2. One-Command Setup
```bash
# Clone repository
git clone https://github.com/Aparna0112/IndicF5-FineTuning.git
cd IndicF5-FineTuning

# Quick setup
chmod +x *.sh
./quick_setup.sh
```

### 3. Prepare Dataset
```bash
python3 prepare_dataset.py
```

### 4. Start Training
```bash
./start_training.sh
```

### 5. Monitor Training
```bash
# In another terminal
./monitor.sh
```

## 📁 Project Structure

```
IndicF5-FineTuning/
├── README.md
├── requirements.txt
├── config.py                 # Training configuration
├── dataset.py               # Malayalam TTS dataset class
├── train.py                 # Main training script
├── prepare_dataset.py       # Dataset preparation
├── inference.py             # Text-to-speech inference
├── setup_runpod.py          # Environment setup
├── start_training.sh        # Training launcher
├── monitor.sh              # System monitor
├── quick_setup.sh          # One-command setup
└── data/
    ├── audio/              # Raw audio files
    ├── processed/          # Processed dataset
    └── dataset_stats.json  # Dataset statistics
```

## 🔧 Configuration

Key settings in `config.py`:

```python
class Config:
    model_name = "ai4bharat/IndicF5"
    language = "malayalam"
    batch_size = 8  # Auto-adjusted based on GPU memory
    learning_rate = 5e-5
    num_epochs = 15
    max_audio_length = 8.0  # seconds
    use_wandb = True
    wandb_project = "indicf5-malayalam-runpod"
```

## 📊 Dataset

- **Source**: `ceymox/MALAYALAM_STT_COMBINED_1_2` from HuggingFace
- **Language**: Malayalam
- **Size**: ~3GB download, varies after processing
- **Quality Filters**: 
  - Audio duration: 1-10 seconds
  - Text length: 5-200 characters
  - Train/Test split: 90/10

## 🏃‍♂️ Training

### Automatic Training
```bash
./start_training.sh
```

### Manual Training
```bash
# Setup environment
python3 setup_runpod.py

# Prepare data
python3 prepare_dataset.py

# Train model
python3 train.py
```

### Training Features
- ✅ Mixed precision training (FP16)
- ✅ Gradient accumulation
- ✅ Automatic checkpoint management
- ✅ WandB logging
- ✅ GPU memory optimization
- ✅ Error handling and recovery

## 📈 Monitoring

### Real-time Monitoring
```bash
# System monitor
./monitor.sh

# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f logs/training.log
```

### WandB Dashboard
- Project: `indicf5-malayalam-runpod`
- Metrics: loss, learning rate, epoch progress
- Login: `wandb login` (optional)

## 🎯 Inference

### Test Trained Model
```bash
python3 inference.py
```

### Custom Inference
```python
from inference import load_finetuned_model, generate_speech, save_audio

# Load model
model = load_finetuned_model()

# Generate speech
malayalam_text = "മലയാളം ടെക്സ്റ്റ് ടു സ്പീച്ച്"
audio = generate_speech(model, malayalam_text)

# Save audio
save_audio(audio, "output.wav")
```

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config.py
batch_size = 2  # or 4
```

**Dataset Download Fails**
```bash
# Manual retry
python3 prepare_dataset.py
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Model Loading Issues**
```bash
# Check model name and HuggingFace access
# Verify: ai4bharat/IndicF5
```

### Performance Tips
- Use A100 40GB for best performance
- Monitor GPU memory with `nvidia-smi`
- Use tmux for long training sessions
- Download checkpoints regularly

## 💾 Checkpoints

Checkpoints are saved to `./checkpoints/indicf5_malayalam/`:
- `best_model.pt` - Best validation loss
- `checkpoint_epoch_N.pt` - Epoch checkpoints
- Automatic cleanup keeps latest 3 checkpoints

### Download Checkpoints
```bash
# Compress before download
tar -czf checkpoints.tar.gz checkpoints/

# Download to local machine
# scp root@runpod-ip:/workspace/IndicF5-FineTuning/checkpoints.tar.gz ./
```

## 📝 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 12.1+ (recommended)
- 16GB+ GPU memory (minimum 8GB)
- 50GB+ storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on RunPod
5. Submit a pull request

## 📄 License

This project is open source. Please respect the licenses of:
- IndicF5 model (AI4Bharat)
- Dataset (ceymox/MALAYALAM_STT_COMBINED_1_2)

## 🆘 Support

- **Issues**: GitHub Issues
- **Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/ceymox/MALAYALAM_STT_COMBINED_1_2)
- **Model**: [IndicF5](https://huggingface.co/ai4bharat/IndicF5)

## 🏆 Results

After training, you should have:
- Fine-tuned IndicF5 model for Malayalam TTS
- Checkpoints for inference
- Training logs and metrics
- Generated speech samples

---

**Happy Training! 🚀**
