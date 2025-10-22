# train_indicf5.py - DEBUG VERSION
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
    print("Malayalam F5-TTS Training - DEBUG MODE")
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
    
    print("\nStarting training (DEBUG MODE - will show all errors)...")
    best_loss = float('inf')
    os.makedirs('./checkpoints/malayalam_f5', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            print(f"\n{'='*60}")
            print(f"Processing Batch {batch_idx}")
            print(f"{'='*60}")
            
            try:
                audios = batch['audios'].to(device)
                text_indices = batch['text_indices'].to(device)
                
                bs = audios.shape[0]
                seq_len = 750
                cond = torch.randn(bs, seq_len, 512, device=device)
                time = torch.rand(bs, device=device)
                
                print(f"Input shapes:")
                print(f"  audios: {audios.shape}")
                print(f"  text_indices: {text_indices.shape}")
                print(f"  cond: {cond.shape}")
                print(f"  time: {time.shape}")
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    x = audios.unsqueeze(1)
                    print(f"  x (with channel): {x.shape}")
                    
                    print(f"Calling model.forward()...")
                    outputs = model(x=x, cond=cond, text=text_indices, time=time)
                    print(f"  outputs: {outputs.shape}")
                    
                    if outputs.shape != x.shape:
                        print(f"Shape mismatch detected!")
                        print(f"  Before adjustment - outputs: {outputs.shape}, x: {x.shape}")
                        
                        if outputs.dim() == 2:
                            outputs = outputs.unsqueeze(1)
                            print(f"  Added channel dimension: {outputs.shape}")
                        elif outputs.shape[1] != 1:
                            outputs = outputs[:, :1, :]
                            print(f"  Took first channel: {outputs.shape}")
                        
                        if outputs.shape[2] != x.shape[2]:
                            min_len = min(outputs.shape[2], x.shape[2])
                            outputs = outputs[:, :, :min_len]
                            x = x[:, :, :min_len]
                            print(f"  Matched length: outputs={outputs.shape}, x={x.shape}")
                    
                    print(f"Computing loss...")
                    loss = torch.nn.functional.mse_loss(outputs, x)
                    print(f"  ✓ Loss: {loss.item():.4f}")
                
                print(f"Backward pass...")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                print(f"✓ Batch {batch_idx} completed successfully!")
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Test only first 3 batches in debug mode
                if batch_idx >= 2:
                    print(f"\n{'='*60}")
                    print(f"DEBUG MODE: Stopping after 3 batches")
                    print(f"{'='*60}")
                    break
                
            except RuntimeError as e:
                print(f"\n❌ RuntimeError in batch {batch_idx}:")
                print(f"   {e}")
                if "out of memory" in str(e):
                    print(f"   -> CUDA Out of Memory")
                    torch.cuda.empty_cache()
                import traceback
                traceback.print_exc()
                continue
                
            except Exception as e:
                print(f"\n❌ Exception in batch {batch_idx}:")
                print(f"   Type: {type(e).__name__}")
                print(f"   Message: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches == 0:
            print("\n❌ No successful batches!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\n{'='*60}")
        print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f} ({num_batches} successful batches)")
        print(f"{'='*60}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'loss': best_loss}, './checkpoints/malayalam_f5/best_model.pt')
            print(f"💾 Saved best: {best_loss:.4f}")
        
        # In debug mode, only run 1 epoch
        print(f"\nDEBUG MODE: Stopping after 1 epoch")
        break
    
    print(f"\n{'='*60}")
    print(f"✅ DEBUG Complete! Best: {best_loss:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
