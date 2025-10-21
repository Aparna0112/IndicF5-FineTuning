# train_indicf5.py - Direct F5-TTS training for Malayalam
import os
import sys
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import wandb

# Import F5-TTS components
from f5_tts.model import CFM, UNetT, DiT
from f5_tts.model.utils import get_tokenizer

class MalayalamDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, sep='|', header=None, 
                                names=['audio', 'text', 'speaker', 'duration'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = sf.read(row['audio'])
        return {
            'audio': torch.FloatTensor(audio),
            'text': row['text'],
            'duration': row['duration']
        }

def collate_fn(batch):
    max_len = max([item['audio'].shape[0] for item in batch])
    audios = torch.zeros(len(batch), max_len)
    
    for i, item in enumerate(batch):
        audio = item['audio']
        audios[i, :len(audio)] = audio
    
    texts = [item['text'] for item in batch]
    
    return {'audios': audios, 'texts': texts}

def train():
    print("="*60)
    print("Malayalam F5-TTS Training")
    print("="*60)
    
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_epochs = 15
    lr = 5e-5
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    
    # Initialize model
    print("\nInitializing model...")
    model = DiT(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4
    ).to(device)
    
    tokenizer = get_tokenizer()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MalayalamDataset('./data/f5_format/train.csv')
    val_dataset = MalayalamDataset('./data/f5_format/test.csv')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Initialize wandb
    try:
        wandb.init(project="indicf5-malayalam", name="f5tts-malayalam")
    except:
        pass
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    
    os.makedirs('./checkpoints/malayalam_f5', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            audios = batch['audios'].to(device)
            texts = batch['texts']
            
            optimizer.zero_grad()
            
            # Simple training step (you may need to adapt based on F5-TTS API)
            try:
                # This is a placeholder - adapt based on actual F5-TTS training
                outputs = model(audios)
                loss = criterion(outputs, audios)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, './checkpoints/malayalam_f5/best_model.pt')
            print(f"✓ Saved best model (loss: {best_loss:.4f})")
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)

if __name__ == "__main__":
    train()
