# train_indicf5.py - Simplified without tokenizer issues
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import wandb

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
        
        # Pad or truncate to fixed length
        target_len = 192000  # 8 seconds at 24kHz
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = torch.nn.functional.pad(
                torch.FloatTensor(audio), 
                (0, target_len - len(audio))
            )
        
        return {
            'audio': torch.FloatTensor(audio) if not isinstance(audio, torch.Tensor) else audio,
            'text': row['text']
        }

def collate_fn(batch):
    audios = torch.stack([item['audio'] for item in batch])
    texts = [item['text'] for item in batch]
    return {'audios': audios, 'texts': texts}

def train():
    print("="*60)
    print("Malayalam F5-TTS Training (Simplified)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2  # Reduced for stability
    num_epochs = 15
    lr = 5e-5
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    
    print("\nInitializing model...")
    model = DiT(
        dim=512,  # Reduced for faster training
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    try:
        wandb.init(project="indicf5-malayalam", name="f5tts-simple")
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
        
        for batch in pbar:
            try:
                audios = batch['audios'].to(device)
                
                optimizer.zero_grad()
                
                # Simple reconstruction loss
                # Note: This is simplified - real F5-TTS training is more complex
                outputs = model(audios)
                loss = torch.nn.functional.mse_loss(outputs, audios)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\nCUDA OOM! Reduce batch_size")
                    torch.cuda.empty_cache()
                    break
                print(f"\nError: {e}")
                continue
        
        if num_batches == 0:
            print("No successful batches!")
            break
            
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, './checkpoints/malayalam_f5/best_model.pt')
            print(f"✓ Saved best model (loss: {best_loss:.4f})")
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    train()
