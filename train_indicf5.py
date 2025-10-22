# train_indicf5.py - ROBUST VERSION
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
        # Filter valid entries
        self.data = self.data[self.data['audio'].notna()]
        
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
                padding = target_len - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            text = str(row['text'])[:256]  # Limit text length
            text_indices = [ord(c) % 256 for c in text]
            
            return {
                'audio': torch.FloatTensor(audio),
                'text_indices': torch.LongTensor(text_indices)
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data
            return {
                'audio': torch.zeros(192000),
                'text_indices': torch.LongTensor([0])
            }

def collate_fn(batch):
    # Filter out None/invalid samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    audios = torch.stack([item['audio'] for item in batch])
    
    # Pad text to same length
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
    batch_size = 4  # Increased batch size
    num_epochs = 15
    lr = 5e-5
    
    print(f"\nDevice: {device}")
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
        drop_last=True  # Drop incomplete batches
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
            if batch is None:
                continue
                
            try:
                audios = batch['audios'].to(device)
                text_indices = batch['text_indices'].to(device)
                
                bs = audios.shape[0]
                cond = torch.randn(bs, 512, device=device)
                time = torch.rand(bs, device=device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # Ensure correct input shape: [batch, channels, length]
                    x = audios.unsqueeze(1) if audios.dim() == 2 else audios
                    
                    outputs = model(
                        x=x,
                        cond=cond,
                        text=text_indices,
                        time=time
                    )
                    
                    # Ensure output and target have same shape
                    if outputs.dim() != x.dim():
                        if outputs.dim() == 2:
                            outputs = outputs.unsqueeze(1)
                    
                    loss = torch.nn.functional.mse_loss(outputs, x)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'batches': num_batches})
                
                if wandb.run and batch_idx % 50 == 0:
                    wandb.log({'train_loss': loss.item(), 'step': epoch * len(train_loader) + batch_idx})
                
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg:
                    print(f"\n⚠️ OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                elif "dimension" in error_msg or "shape" in error_msg:
                    print(f"\n⚠️ Shape error at batch {batch_idx}: {error_msg}")
                    continue
                else:
                    print(f"\n❌ Runtime error: {error_msg}")
                    continue
            except Exception as e:
                print(f"\n❌ Error at batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            print("\n❌ No successful batches!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\n✅ Epoch {epoch+1}: Loss={avg_loss:.4f} ({num_batches} batches)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, './checkpoints/malayalam_f5/best_model.pt')
            print(f"💾 Saved best: {best_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, f'./checkpoints/malayalam_f5/epoch_{epoch+1}.pt')
        
        if wandb.run:
            wandb.log({'epoch': epoch, 'avg_loss': avg_loss, 'best_loss': best_loss})
    
    print("\n" + "="*60)
    print(f"✅ Training complete! Best loss: {best_loss:.4f}")
    print("="*60)
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
