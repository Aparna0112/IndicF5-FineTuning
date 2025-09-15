# config.py
import torch

class Config:
    # Model settings
    model_name = "ai4bharat/IndicF5"
    language = "malayalam"
    
    # Training settings
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 20
    warmup_steps = 500
    max_length = 200
    
    # Audio settings
    sample_rate = 24000
    max_audio_length = 10.0  # seconds
    
    # Paths
    data_dir = "./data/processed"
    output_dir = "./indicf5_malayalam_finetuned"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    logging_steps = 50
    save_steps = 500
    eval_steps = 500
