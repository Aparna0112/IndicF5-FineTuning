# train_indicf5.py - CORRECT INITIALIZATION
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import numpy as np
import torchaudio

from f5_tts.model import CFM, DiT
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
        
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(
                torch.FloatTensor(audio), sr, self.target_sr
            ).numpy()
        
        max_len = self.target_sr * 10
        if len(audio) > max_len:
            audio = audio[:max_len]
        
        text = str(row['text'])
        
        return {'audio': torch.FloatTensor(audio), 'text': text}

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    max_len = len(batch[0]['audio'])
    audios = []
    for item in batch:
        audio = item['audio']
        if len(audio) < max_len:
            audio = torch.nn.functional.pad(audio, (0, max_len - len(audio)))
        audios.append(audio)
    
    audios = torch.stack(audios)
    texts = [item['text'] for item in batch]
    
    return {'audios': audios, 'texts': texts}

def train():
    print("="*60)
    print("Malayalam F5-TTS Training")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    num_epochs = 15
    lr = 7.5e-5
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    
    print("\nInitializing model...")
    
    # Mel spectrogram
    mel_spec = MelSpec(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000
    )
    
    # DiT transformer
    transformer = DiT(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4
    )
    
    # CFM with CORRECT parameters
    model = CFM(
        transformer=transformer,
        sigma=0.0,
        mel_spec_module=mel_spec,
        odeint_kwargs={'method': 'euler'},
        audio_drop_prob=0.3,
        cond_drop_prob=0.2
    ).to(device)
    
    print("✓ Model initialized successfully")
    
    print("\nLoading dataset...")
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
    
    print(f"Total batches: {len(train_loader)}")
    print("\nStarting training...\n")
    
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
                
                optimizer.zero_grad()
                
                # CFM handles mel conversion internally now
                loss = model(audios, texts)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})
                
                if batch_idx == 0:
                    print(f"\n✓ First batch successful! Loss: {loss.item():.4f}")
                
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"\n  Batch {batch_idx}: Loss={loss.item():.4f}, Avg={total_loss/num_batches:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    if batch_idx < 3:
                        print(f"\n❌ Error: {e}")
                    continue
            except Exception as e:
                if batch_idx < 3:
                    print(f"\n❌ Unexpected: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if num_batches == 0:
            print("\n❌ No successful batches!")
            break
        
        avg_loss = total_loss / num_batches
        success_rate = (num_batches / len(train_loader)) * 100
        
        print(f"\n{'='*60}")
        print(f"✅ Epoch {epoch+1}/{num_epochs}")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Success: {num_batches}/{len(train_loader)} ({success_rate:.1f}%)")
        print(f"{'='*60}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, './checkpoints/malayalam_cfm/best_model.pt')
            print(f"💾 Saved best model: {best_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss
            }, f'./checkpoints/malayalam_cfm/epoch_{epoch+1}.pt')
            print(f"💾 Saved epoch {epoch+1} checkpoint")
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Checkpoints: ./checkpoints/malayalam_cfm/")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
