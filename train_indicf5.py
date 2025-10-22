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
        self.data = pd.read_csv(csv_path, sep='|', header=None, 
                                names=['audio', 'text', 'speaker', 'duration'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = sf.read(row['audio'])
        
        target_len = 192000
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            padding = target_len - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Convert text to character indices (Long type)
        text = row['text']
        text_indices = [ord(c) % 256 for c in text[:256]]  # Limit to 256 chars
        
        return {
            'audio': torch.FloatTensor(audio),
            'text_indices': torch.LongTensor(text_indices),
            'text': text
        }

def collate_fn(batch):
    audios = torch.stack([item['audio'] for item in batch])
    
    # Pad text indices to same length
    max_text_len = max(len(item['text_indices']) for item in batch)
    text_indices_list = []
    
    for item in batch:
        indices = item['text_indices']
        # Pad with zeros
        padded = torch.nn.functional.pad(indices, (0, max_text_len - len(indices)), value=0)
        text_indices_list.append(padded)
    
    text_indices = torch.stack(text_indices_list)
    texts = [item['text'] for item in batch]
    
    return {
        'audios': audios,
        'text_indices': text_indices,
        'texts': texts
    }

def train():
    print("="*60)
    print("Malayalam F5-TTS Training")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
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
    val_dataset = MalayalamDataset('./data/f5_format/test.csv')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
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
                
                # Verify text_indices are Long type
                assert text_indices.dtype == torch.long, f"Expected Long, got {text_indices.dtype}"
                
                batch_size_actual = audios.shape[0]
                cond = torch.randn(batch_size_actual, 512).to(device)
                time = torch.rand(batch_size_actual).to(device)
                
                optimizer.zero_grad()
                
                # Pass text_indices directly - DiT will handle embedding internally
                outputs = model(
                    x=audios.unsqueeze(1),
                    cond=cond,
                    text=text_indices,  # Pass indices, not embeddings
                    time=time
                )
                
                target = audios.unsqueeze(1)
                loss = torch.nn.functional.mse_loss(outputs, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if wandb.run and batch_idx % 10 == 0:
                    wandb.log({
                        'train_loss': loss.item(),
                        'step': epoch * len(train_loader) + batch_idx
                    })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ CUDA OOM at batch {batch_idx}! Skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\n❌ Error at batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                print(f"\n❌ Unexpected error at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches == 0:
            print("\n❌ No successful batches in this epoch!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\n✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} ({num_batches}/{len(train_loader)} batches)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, './checkpoints/malayalam_f5/best_model.pt')
            print(f"💾 Saved best model (loss: {best_loss:.4f})")
        
        # Save every epoch
        torch.save(checkpoint, f'./checkpoints/malayalam_f5/checkpoint_epoch_{epoch}.pt')
        
        if wandb.run:
            wandb.log({'epoch': epoch, 'avg_loss': avg_loss, 'best_loss': best_loss})
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: ./checkpoints/malayalam_f5/")
    print("="*60)
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    train()
