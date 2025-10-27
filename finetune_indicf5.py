import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModel

print("="*60)
print("Fine-tuning IndicF5 for Malayalam")
print("="*60)

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
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    num_epochs = 10
    lr = 1e-5
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    print("Loading IndicF5...")
    
    try:
        model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
        
        if hasattr(model, 'model'):
            f5_model = model.model
        else:
            f5_model = model
        
        f5_model = f5_model.to(device)
        f5_model.train()
        print("✓ IndicF5 loaded!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Using F5-TTS instead...")
        
        from f5_tts.model import CFM, DiT
        from f5_tts.model.modules import MelSpec
        
        mel_spec = MelSpec(n_fft=1024, hop_length=256, win_length=1024,
                          n_mel_channels=100, target_sample_rate=24000)
        
        transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2,
                         text_dim=512, conv_layers=4)
        
        f5_model = CFM(transformer=transformer, sigma=0.0,
                      mel_spec_module=mel_spec,
                      odeint_kwargs={'method': 'euler'},
                      audio_drop_prob=0.3, cond_drop_prob=0.2).to(device)
    
    print("Loading dataset...")
    train_dataset = MalayalamDataset('../IndicF5-FineTuning/data/f5_format/train.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4,
                             pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(f5_model.parameters(), lr=lr, weight_decay=0.01)
    
    print(f"Batches: {len(train_loader)}")
    print("Starting training...\\n")
    
    best_loss = float('inf')
    os.makedirs('../checkpoints/indicf5_malayalam', exist_ok=True)
    
    for epoch in range(num_epochs):
        f5_model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                audios = batch['audios'].to(device)
                texts = batch['texts']
                
                optimizer.zero_grad()
                
                output = f5_model(audios, texts)
                loss = output[0] if isinstance(output, tuple) else output
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(f5_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                 'avg': f'{total_loss/num_batches:.4f}'})
                
                if batch_idx == 0:
                    print(f"\\n✓ First batch! Loss: {loss.item():.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                continue
        
        if num_batches == 0:
            break
        
        avg_loss = total_loss / num_batches
        print(f"\\n✅ Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model_state_dict': f5_model.state_dict(),
                       'loss': best_loss}, 
                      '../checkpoints/indicf5_malayalam/best_model.pt')
            print(f"💾 Saved! Loss: {best_loss:.4f}")
    
    print(f"\\n✅ Complete! Best: {best_loss:.4f}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\\n⚠️ Interrupted")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
