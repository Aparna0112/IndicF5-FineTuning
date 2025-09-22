# inference.py - Corrected version
import torch
from transformers import AutoModel
import soundfile as sf
import numpy as np
import os
import glob

def load_finetuned_model(checkpoint_path=None):
    """Load the fine-tuned model"""
    
    # Auto-find latest checkpoint if path not provided
    if checkpoint_path is None:
        checkpoint_dir = "./checkpoints/indicf5_malayalam"
        if os.path.exists(checkpoint_dir):
            # Look for best model first
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            if os.path.exists(best_model_path):
                checkpoint_path = best_model_path
                print(f"Using best model: {checkpoint_path}")
            else:
                # Find latest checkpoint
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
                if checkpoint_files:
                    # Sort by epoch number
                    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = checkpoint_files[-1]  # Latest checkpoint
                    print(f"Using latest checkpoint: {checkpoint_path}")
                else:
                    raise FileNotFoundError("No checkpoints found in ./checkpoints/indicf5_malayalam/")
        else:
            raise FileNotFoundError("Checkpoint directory not found: ./checkpoints/indicf5_malayalam/")
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load base model
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    return model

def generate_speech(model, text, ref_audio_path=None, ref_text=None):
    """Generate speech from Malayalam text"""
    
    # Use a reference audio if provided, otherwise use default
    if ref_audio_path is None:
        # Look for any audio file in the processed data
        audio_dir = "./data/audio"
        if os.path.exists(audio_dir):
            audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
            if audio_files:
                ref_audio_path = audio_files[0]  # Use first available audio
                print(f"Using reference audio: {ref_audio_path}")
            else:
                print("Warning: No reference audio found, using None")
        else:
            print("Warning: Audio directory not found, using None reference")
    
    if ref_text is None:
        # Use corresponding text from your dataset or default
        ref_text = "നമസ്കാരം"  # Example Malayalam text
    
    try:
        # Generate audio using IndicF5
        # Note: The exact API might vary based on IndicF5 implementation
        if ref_audio_path:
            audio = model.generate(
                text=text,
                ref_audio=ref_audio_path,
                ref_text=ref_text
            )
        else:
            # Fallback if no reference audio
            audio = model.generate(text=text)
        
        return audio
    
    except Exception as e:
        print(f"Error generating speech: {e}")
        print("This might be due to IndicF5 API changes. Please check the model documentation.")
        return None

def save_audio(audio, output_path, sample_rate=24000):
    """Save generated audio to file"""
    if audio is None:
        print("No audio to save")
        return False
    
    try:
        # Ensure audio is in the right format
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        # Normalize if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        sf.write(output_path, audio, sample_rate)
        print(f"Generated audio saved as: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

# Example usage
if __name__ == "__main__":
    try:
        # Load model (will auto-find latest checkpoint)
        model = load_finetuned_model()
        
        # Test Malayalam text
        malayalam_text = "മലയാളം ടെക്സ്റ്റ് ടു സ്പീച്ച് മോഡൽ"
        print(f"Generating speech for: {malayalam_text}")
        
        # Generate speech
        audio = generate_speech(model, malayalam_text)
        
        # Save generated audio
        output_path = "generated_malayalam.wav"
        success = save_audio(audio, output_path)
        
        if success:
            print("✓ Inference completed successfully!")
        else:
            print("✗ Inference failed!")
            
    except Exception as e:
        print(f"Error in inference: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have trained checkpoints in ./checkpoints/indicf5_malayalam/")
        print("2. Check if the model was trained successfully")
        print("3. Verify IndicF5 model API (it might have changed)")
