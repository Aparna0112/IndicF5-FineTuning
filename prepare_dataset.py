# prepare_dataset.py
import os
from datasets import load_dataset, Dataset, Audio
import soundfile as sf
import numpy as np
import torch
from pathlib import Path

def prepare_malayalam_tts_data():
    # Load your TTS dataset
    print("Loading Malayalam TTS dataset...")
    dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2")
    
    # Create directories
    os.makedirs("./data/audio", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    
    def process_sample(example, idx):
        # Extract audio data
        audio_array = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        text = example['transcription']
        
        # Save audio file
        audio_path = f"./data/audio/malayalam_{idx:06d}.wav"
        sf.write(audio_path, audio_array, sample_rate)
        
        return {
            'audio_path': audio_path,
            'text': text,
            'sample_rate': sample_rate,
            'duration': len(audio_array) / sample_rate
        }
    
    # Process all samples
    processed_data = []
    for idx, example in enumerate(dataset['train']):
        processed_sample = process_sample(example, idx)
        processed_data.append(processed_sample)
        
        if idx % 100 == 0:
            print(f"Processed {idx} samples...")
    
    # Create new dataset
    processed_dataset = Dataset.from_list(processed_data)
    
    # Split into train/validation
    train_test = processed_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Save processed dataset
    train_test.save_to_disk("./data/processed")
    print(f"Saved {len(train_test['train'])} training and {len(train_test['test'])} validation samples")
    
    return train_test

if __name__ == "__main__":
    dataset = prepare_malayalam_tts_data()

# explore_dataset.py - Run this first to understand your data
def explore_dataset():
    """Explore the structure and content of your Malayalam TTS dataset"""
    print("Loading dataset for exploration...")
    dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2")
    
    print(f"\nDataset structure:")
    print(f"- Splits: {list(dataset.keys())}")
    print(f"- Number of samples: {len(dataset['train'])}")
    print(f"- Features: {dataset['train'].features}")
    
    # Sample data
    sample = dataset['train'][0]
    print(f"\nFirst sample:")
    print(f"- Index: {sample['index']}")
    print(f"- Transcription: {sample['transcription']}")
    print(f"- Audio shape: {sample['audio']['array'].shape}")
    print(f"- Sample rate: {sample['audio']['sampling_rate']}")
    print(f"- Duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f} seconds")
    
    # Dataset statistics
    durations = []
    text_lengths = []
    
    print("\nAnalyzing dataset statistics...")
    for i, item in enumerate(dataset['train']):
        duration = len(item['audio']['array']) / item['audio']['sampling_rate']
        durations.append(duration)
        text_lengths.append(len(item['transcription']))
        
        if i >= 100:  # Analyze first 100 samples for quick stats
            break
    
    print(f"\nAudio Statistics (first 100 samples):")
    print(f"- Average duration: {np.mean(durations):.2f} seconds")
    print(f"- Min duration: {np.min(durations):.2f} seconds")
    print(f"- Max duration: {np.max(durations):.2f} seconds")
    
    print(f"\nText Statistics (first 100 samples):")
    print(f"- Average text length: {np.mean(text_lengths):.0f} characters")
    print(f"- Min text length: {np.min(text_lengths)} characters")
    print(f"- Max text length: {np.max(text_lengths)} characters")
    
    # Show sample texts
    print(f"\nSample Malayalam texts:")
    for i in range(5):
        sample = dataset['train'][i]
        print(f"{i+1}. {sample['transcription']}")

# Run exploration first
# explore_dataset()
