# train_indicf5.py - FINAL FIXED VERSION
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
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': row['text']
        }

def collate_fn(batch):
    audios = torch.stack([item['audio'] for item in batch])
    texts = [item['text'] for item in batch]
    
    max_text_len = max(len(t) for t in texts)
    text_embeds = []
    
    for text in texts:
        chars = [ord(c) % 256 for c in text]
        chars = chars + [0] * (max_text_len - len(chars))
        text_embeds.append(chars[:256])
    
    # FIXED: Explicitly create LongTensor
    text_tensor = torch.LongTensor(text_embeds)
    
    return {
        'audios': audios,
        'texts': texts,
        'text_embeds': text_tensor
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
    
    text_embed_layer = torch.nn.Embedding(256, 256).to(device)
    
    print("Loading datasets...")
    train_dataset = MalayalamDataset('./data/f5_format/train.csv')
    val_dataset = MalayalamDataset('./data/f5_format/test.csv')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(text_embed_layer.parameters()), 
        lr=lr
    )
    
    try:
        wandb.init(project="indicf5-malayalam", name="f5tts-malayalam")
    except:
        print("WandB not initialized")
    
    print("\nStarting training...")
    best_loss = float('inf')
    os.makedirs('./checkpoints/malayalam_f5', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        text_embed_layer.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                audios = batch['audios'].to(device)
                # FIXED: Ensure Long type before sending to device
                text_indices = batch['text_embeds'].to(device)
                
                # Verify type
                if text_indices.dtype != torch.long:
                    text_indices = text_indices.long()
                
                text_embeds = text_embed_layer(text_indices)
                
                batch_size_actual = audios.shape[0]
                cond = torch.randn(batch_size_actual, 512).to(device)
                time = torch.rand(batch_size_actual).to(device)
                
                optimizer.zero_grad()
                
                outputs = model(
                    x=audios.unsqueeze(1),
                    cond=cond,
                    text=text_embeds,
                    time=time
                )
                
                target = audios.unsqueeze(1)
                loss = torch.nn.functional.mse_loss(outputs, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(text_embed_layer.parameters()), 
                    1.0
                )
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if wandb.run:
                    wandb.log({'train_loss': loss.item(), 'step': epoch * len(train_loader) + batch_idx})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA OOM at batch {batch_idx}! Clearing cache...")
                    torch.cuda.empty_cache()
                    continue  # Skip this batch and continue
                else:
                    print(f"\nError at batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"\nUnexpected error at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches == 0:
            print("❌ No successful batches in this epoch!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\n✓ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} ({num_batches} batches)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'text_embed_state_dict': text_embed_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, './checkpoints/malayalam_f5/best_model.pt')
            print(f"💾 Saved best model (loss: {best_loss:.4f})")
        
        torch.save(checkpoint, f'./checkpoints/malayalam_f5/checkpoint_epoch_{epoch}.pt')
        
        if wandb.run:
            wandb.log({'epoch': epoch, 'avg_loss': avg_loss})
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print("="*60)
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    train()
