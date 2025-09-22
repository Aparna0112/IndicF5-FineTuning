import torch
import os

class Config:
    # Model settings
    model_name = "ai4bharat/IndicF5"  # Corrected: Use base IndicF5 model
    language = "malayalam"
    
    # Training settings - adjusted for RunPod
    # Safer batch size calculation with fallback
    @staticmethod
    def get_batch_size():
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory > 20e9:  # > 20GB
                    return 8
                elif total_memory > 10e9:  # > 10GB
                    return 4
                else:
                    return 2
            else:
                return 2
        except:
            return 2  # Safe fallback
    
    batch_size = get_batch_size()
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
    
    # Safety settings
    fp16 = True  # Use mixed precision training
    gradient_checkpointing = True  # Save memory
    dataloader_drop_last = True  # Avoid size mismatch issues
