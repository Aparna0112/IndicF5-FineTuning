# prepare_f5_data.py - Fixed version
import os
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

print("Loading processed dataset...")
dataset = load_from_disk("./data/processed")

os.makedirs("./data/f5_format", exist_ok=True)

def create_metadata(split_name):
    print(f"\nProcessing {split_name} split...")
    split_data = dataset[split_name]
    
    metadata = []
    for idx, item in enumerate(tqdm(split_data)):
        # F5 format: audio_path|text|speaker|duration
        metadata.append([
            item['audio_path'],
            item['text'],
            'malayalam_speaker',
            item['duration']
        ])
    
    # Save as CSV without index
    df = pd.DataFrame(metadata, columns=['audio', 'text', 'speaker', 'duration'])
    csv_path = f"./data/f5_format/{split_name}.csv"
    
    # Save with pipe delimiter, no header, no index
    df.to_csv(csv_path, sep='|', header=False, index=False)
    
    print(f"Saved {len(metadata)} samples to {csv_path}")
    
    # Show first few lines
    print(f"Sample lines from {csv_path}:")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {line.strip()}")
    
    return df

# Create metadata for both splits
train_df = create_metadata('train')
test_df = create_metadata('test')

print("\n" + "="*60)
print("✓ F5-TTS format data prepared!")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print("="*60)
