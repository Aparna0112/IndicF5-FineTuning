# prepare_f5_data.py - Prepare Malayalam data for F5-TTS training
import os
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

print("Loading processed dataset...")
dataset = load_from_disk("./data/processed")

# Create metadata directory
os.makedirs("./data/f5_format", exist_ok=True)

def create_metadata(split_name):
    print(f"\nProcessing {split_name} split...")
    split_data = dataset[split_name]
    
    metadata = []
    for idx, item in enumerate(tqdm(split_data)):
        # F5-TTS expects: audio_path|text|speaker_id|duration
        metadata.append({
            'audio': item['audio_path'],
            'text': item['text'],
            'speaker': 'malayalam_speaker',
            'duration': item['duration']
        })
    
    # Save as CSV
    df = pd.DataFrame(metadata)
    csv_path = f"./data/f5_format/{split_name}.csv"
    df.to_csv(csv_path, index=False, sep='|', header=False)
    
    print(f"Saved {len(metadata)} samples to {csv_path}")
    return df

# Create metadata for both splits
train_df = create_metadata('train')
test_df = create_metadata('test')

print("\n" + "="*60)
print("✓ F5-TTS format data prepared!")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print("="*60)
