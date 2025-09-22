import os
from datasets import load_dataset, Dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
import json

def prepare_malayalam_tts_data():
    """Prepare Malayalam TTS dataset optimized for RunPod"""
    print("Loading Malayalam TTS dataset from HuggingFace...")
    
    try:
        # Load dataset
        dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2", split="train")
        print(f"Loaded {len(dataset)} samples")
        
        # Create directories
        os.makedirs("./data/audio", exist_ok=True)
        os.makedirs("./data/processed", exist_ok=True)
        
        # Filter and process data
        processed_data = []
        skipped = 0
        
        for idx, example in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Extract data
                audio_array = example['audio']['array']
                sample_rate = example['audio']['sampling_rate']
                text = example['transcription'].strip()
                
                # Quality filters
                duration = len(audio_array) / sample_rate
                if duration < 1.0 or duration > 10.0:
                    skipped += 1
                    continue
                
                if len(text) < 5 or len(text) > 200:
                    skipped += 1
                    continue
                
                # Save audio file
                audio_path = f"./data/audio/malayalam_{idx:06d}.wav"
                sf.write(audio_path, audio_array, sample_rate)
                
                processed_data.append({
                    'audio_path': audio_path,
                    'text': text,
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'text_length': len(text)
                })
                
                # Progress update
                if idx % 500 == 0:
                    print(f"Processed {idx} samples, skipped {skipped}")
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                skipped += 1
                continue
        
        print(f"Successfully processed {len(processed_data)} samples, skipped {skipped}")
        
        # Create dataset
        processed_dataset = Dataset.from_list(processed_data)
        
        # Split data
        train_test = processed_dataset.train_test_split(test_size=0.1, seed=42)
        
        # Save processed dataset
        train_test.save_to_disk("./data/processed")
        
        # Save statistics
        stats = {
            'total_samples': len(processed_data),
            'train_samples': len(train_test['train']),
            'test_samples': len(train_test['test']),
            'avg_duration': np.mean([item['duration'] for item in processed_data]),
            'avg_text_length': np.mean([item['text_length'] for item in processed_data])
        }
        
        with open('./data/dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset prepared successfully!")
        print(f"Train samples: {len(train_test['train'])}")
        print(f"Test samples: {len(train_test['test'])}")
        
        return train_test
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise

if __name__ == "__main__":
    dataset = prepare_malayalam_tts_data()
