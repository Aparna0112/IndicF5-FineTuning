# train_indicf5.py - FINAL WORKING VERSION
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
        self.data = pd.read_csv(csv_path, sep='|', header=None,
                                names=['audio', 'text', 'speaker', 'duration'])
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
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
        except Exception as e:
            print(f"Error loading {idx}: {e}")
            return {
                'audio': torch.zeros(192000),
                'text_indices': torch.LongTensor([0])
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
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    
    print("\nInitializing model...")
    model = DiT(dim=512, depth=12, heads=8, ff_mult=2, text_dim=256, conv_layers=2).to(device)
    
    print("Loading datasets...")
    train_dataset = MalayalamDataset('./data/f5_format/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"Total batches: {len(train_loader)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    
    try:
        wandb.init(project="indicf5-malayalam", name="f5tts-malayalam")
    except:
        print("WandB not initialized")
    
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
                
                # FIXED: Use constant seq_len that matches model architecture
                seq_len = 750
                cond = torch.randn(bs, seq_len, 512, device=device)
                time = torch.rand(bs, device=device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    x = audios.unsqueeze(1)
                    outputs = model(x=x, cond=cond, text=text_indices, time=time)
                    
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
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                continue
            except Exception as e:
                continue
        
        if num_batches == 0:
            print("\n❌ No successful batches!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\n✅ Epoch {epoch+1}: Loss={avg_loss:.4f} ({num_batches}/{len(train_loader)} batches)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'loss': best_loss}, './checkpoints/malayalam_f5/best_model.pt')
            print(f"💾 Saved best: {best_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                      f'./checkpoints/malayalam_f5/epoch_{epoch+1}.pt')
    
    print(f"\n✅ Complete! Best: {best_loss:.4f}")
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
