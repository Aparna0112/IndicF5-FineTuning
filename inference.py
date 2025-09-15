# inference.py
import torch
from transformers import AutoModel
import soundfile as sf
import numpy as np

def load_finetuned_model(checkpoint_path):
    """Load the fine-tuned model"""
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def generate_speech(model, text, ref_audio_path=None, ref_text=None):
    """Generate speech from Malayalam text"""
    
    # Use a reference audio if provided, otherwise use default
    if ref_audio_path is None:
        # You can use one of your training samples as reference
        ref_audio_path = "./data/audio/malayalam_000001.wav"
    
    if ref_text is None:
        # Use corresponding text from your dataset
        ref_text = "നമസ്കാരം"  # Example Malayalam text
    
    # Generate audio
    audio = model(
        text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text
    )
    
    return audio

# Example usage
if __name__ == "__main__":
    model = load_finetuned_model("./indicf5_malayalam_finetuned/checkpoint_epoch_10.pt")
    
    malayalam_text = "മലയാളം ടെക്സ്റ്റ് ടു സ്പീച്ച് മോഡൽ"
    audio = generate_speech(model, malayalam_text)
    
    # Save generated audio
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    sf.write("generated_malayalam.wav", np.array(audio, dtype=np.float32), samplerate=24000)
    print("Generated audio saved as 'generated_malayalam.wav'")
