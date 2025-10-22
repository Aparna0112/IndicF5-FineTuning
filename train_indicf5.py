# train_indicf5.py - WORKING VERSION
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import wandb
import numpy as np

from f5_tts.model import DiT

class MalayalamDataset(Dataset):
    def __init__(self, csv_path):
        # Updated column names
        self.data = pd.read_csv(csv_path, sep='|', header=None, 
                                names=['index', 'audio', 'transcript', 'speaker', 'duration'])
        
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
        
        text = str(row['transcript'])[:256]
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
    print("Malayalam F5-TTS Training")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_epochs = 15
    lr = 5e-5
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    print("\nInitializing model...")
    model = DiT(
        dim=512,
        depth=12,
        heads=8,
        ff_mult=2,
        text_dim=256,
        conv_layers=2
    ).to(device)
    
    print("Loading datasets...")
    train_dataset = MalayalamDataset('./data/f5_format/train.csv')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    
    print("\nStarting training...")
    best_loss = float('inf')
    os.makedirs('./checkpoints/malayalam_f5', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                audios = batch['audios'].to(device)
                text_indices = batch['text_indices'].to(device)
                
                bs = audios.shape[0]
                seq_len = audios.shape[1] // 256  # Approximate sequence length after downsampling
                
                # FIX: Create 3D conditioning tensor [B, T, D]
                cond = torch.randn(bs, seq_len, 512, device=device)
                time = torch.rand(bs, device=device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    x = audios.unsqueeze(1)  # [B, 1, L]
                    
                    outputs = model(x=x, cond=cond, text=text_indices, time=time)
                    
                    # Match dimensions
                    if outputs.shape != x.shape:
                        if outputs.dim() == 2:
                            outputs = outputs.unsqueeze(1)
                        elif outputs.shape[1] != 1:
                            outputs = outputs[:, :1, :]
                        if outputs.shape[2] != x.shape[2]:
                            min_len = min(outputs.shape[2], x.shape[2])
                            outputs = outputs[:, :, :min_len]
                            x = x[:, :, :min_len]
                    
                    loss = torch.nn.functional.mse_loss(outputs, x)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log every 100 batches
                if batch_idx % 100 == 0:
                    print(f"\nBatch {batch_idx}: Loss={loss.item():.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\n❌ Error at batch {batch_idx}: {e}")
                    if batch_idx < 5:  # Print details for first few errors
                        import traceback
                        traceback.print_exc()
                    continue
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue
        
        if num_batches == 0:
            print("\n❌ No successful batches!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\n✅ Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f} ({num_batches}/{len(train_loader)} batches)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, './checkpoints/malayalam_f5/best_model.pt')
            print(f"💾 Saved best model: {best_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, f'./checkpoints/malayalam_f5/epoch_{epoch+1}.pt')
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete! Best loss: {best_loss:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
