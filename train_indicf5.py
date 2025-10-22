# train_indicf5.py - PROPER F5-TTS TRAINING
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torchaudio

# Use F5-TTS components properly
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.modules import MelSpec

class MalayalamDataset(Dataset):
    def __init__(self, csv_path, target_sample_rate=24000):
        self.data = pd.read_csv(csv_path, sep='|', header=None,
                                names=['audio', 'text', 'speaker', 'duration'])
        self.target_sr = target_sample_rate
        print(f"Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = sf.read(row['audio'])
        
        # Resample to 24kHz if needed
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(
                torch.FloatTensor(audio), sr, self.target_sr
            ).numpy()
        
        # Limit to 10 seconds
        max_len = self.target_sr * 10
        if len(audio) > max_len:
            audio = audio[:max_len]
        
        text = str(row['text'])
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': text,
            'duration': len(audio) / self.target_sr
        }

def collate_fn(batch):
    # Sort by length for efficient padding
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    
    # Pad audios
    max_len = len(batch[0]['audio'])
    audios = []
    for item in batch:
        audio = item['audio']
        if len(audio) < max_len:
            audio = torch.nn.functional.pad(audio, (0, max_len - len(audio)))
        audios.append(audio)
    
    audios = torch.stack(audios)
    texts = [item['text'] for item in batch]
    durations = [item['duration'] for item in batch]
    
    return {
        'audios': audios,
        'texts': texts,
        'durations': durations
    }

def train():
    print("="*60)
    print("Malayalam F5-TTS Training (Proper Method)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    num_epochs = 15
    lr = 7.5e-5
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    
    # Initialize mel spectrogram converter
    mel_spec = MelSpec(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        normalize=False
    ).to(device)
    
    # Initialize CFM model (not raw DiT)
    print("\nInitializing CFM model...")
    model = CFM(
        transformer=dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4
        ),
        mel_spec=mel_spec
    ).to(device)
    
    print("Loading dataset...")
    train_dataset = MalayalamDataset('./data/f5_format/train.csv')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    print(f"\nTotal batches: {len(train_loader)}")
    print("\nStarting training...")
    
    best_loss = float('inf')
    os.makedirs('./checkpoints/malayalam_cfm', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                audios = batch['audios'].to(device)
                texts = batch['texts']
                
                # Convert to mel spectrogram
                with torch.no_grad():
                    mels = mel_spec(audios)
                
                optimizer.zero_grad()
                
                # CFM training
                loss, _ = model(mels, texts)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if batch_idx == 0:
                    print(f"\n  First batch successful! Loss: {loss.item():.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\n❌ Error: {e}")
                    continue
        
        if num_batches == 0:
            print("\n❌ No successful batches!")
            break
        
        avg_loss = total_loss / num_batches
        print(f"\n✅ Epoch {epoch+1}: Loss={avg_loss:.4f} ({num_batches} batches)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss
            }, './checkpoints/malayalam_cfm/best_model.pt')
            print(f"💾 Saved best: {best_loss:.4f}")
    
    print(f"\n✅ Training complete! Best: {best_loss:.4f}")

if __name__ == "__main__":
    train()
