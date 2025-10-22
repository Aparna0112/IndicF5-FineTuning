# train_indicf5.py - NO ERROR CATCHING VERSION
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import numpy as np

from f5_tts.model import DiT

class MalayalamDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep='|', header=None,
                                names=['audio', 'text', 'speaker', 'duration'])
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = sf.read(row['audio'])
        target_len = 192000
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        
        text = str(row['text'])[:256]
        text_indices = [ord(c) % 256 for c in text]
        
        return {
            'audio': torch.FloatTensor(audio),
            'text_indices': torch.LongTensor(text_indices)
        }

def collate_fn(batch):
    audios = torch.stack([item['audio'] for item in batch])
    max_len = max(len(item['text_indices']) for item in batch)
    text_list = []
    for item in batch:
        indices = item['text_indices']
        if len(indices) < max_len:
            indices = torch.cat([indices, torch.zeros(max_len - len(indices), dtype=torch.long)])
        text_list.append(indices)
    text_indices = torch.stack(text_list)
    return {'audios': audios, 'text_indices': text_indices}

def train():
    print("="*60)
    print("Malayalam F5-TTS Training - Will crash on first error")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    print(f"\nDevice: {device}")
    
    model = DiT(dim=512, depth=12, heads=8, ff_mult=2, text_dim=256, conv_layers=2).to(device)
    
    train_dataset = MalayalamDataset('./data/f5_format/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\nStarting training (will show first error)...")
    
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx}")
        print(f"{'='*60}")
        
        audios = batch['audios'].to(device)
        text_indices = batch['text_indices'].to(device)
        
        print(f"Shapes:")
        print(f"  audios: {audios.shape}")
        print(f"  text: {text_indices.shape}")
        
        bs = audios.shape[0]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            x = audios.unsqueeze(1)
            print(f"  x: {x.shape}")
            
            # Try seq_len = 1
            cond = torch.randn(bs, 1, 512, device=device)
            print(f"  cond: {cond.shape}")
            
            time = torch.rand(bs, device=device)
            print(f"  time: {time.shape}")
            
            print(f"\nCalling model...")
            outputs = model(x=x, cond=cond, text=text_indices, time=time)
            print(f"  outputs: {outputs.shape}")
            
            loss = torch.nn.functional.mse_loss(outputs, x)
            print(f"  loss: {loss.item():.4f}")
        
        print(f"\nBackward...")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✓ Batch {batch_idx} SUCCESS!")
        
        # Just test first batch
        break
    
    print(f"\n✅ First batch worked!")

if __name__ == "__main__":
    train()
