# dataset.py
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import numpy as np
from datasets import load_from_disk

class MalayalamTTSDataset(Dataset):
    def __init__(self, data_path, split="train", config=None):
        self.config = config
        self.dataset = load_from_disk(data_path)[split]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and process audio
        audio_path = item['audio_path']
        audio, sr = sf.read(audio_path)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Truncate or pad audio
        max_samples = int(self.config.max_audio_length * self.config.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        return {
            'text': item['text'],
            'audio': torch.FloatTensor(audio),
            'audio_path': audio_path,
            'duration': item['duration']
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    texts = [item['text'] for item in batch]
    audios = torch.stack([item['audio'] for item in batch])
    
    return {
        'texts': texts,
        'audios': audios,
        'batch_size': len(batch)
    }
