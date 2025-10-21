import os
from datasets import load_dataset, Dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
import json

def prepare_malayalam_tts_data():
    print("="*60)
    print("Malayalam TTS Dataset Preparation")
    print("="*60)
    
    os.makedirs("./data/audio", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    
    print("\nLoading dataset from HuggingFace...")
    dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2", split="train")
    print(f"✓ Loaded {len(dataset)} samples")
    
    print("\nProcessing audio samples...")
    processed_data = []
    skipped = 0
    
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        try:
            example = dataset[idx]
            
            # Get audio array directly
            audio_array = np.array(example['audio']['array'])
            sr = example['audio']['sampling_rate']
            
            # Get text
            text = example['transcription'].strip()
            
            # Filters
            duration = len(audio_array) / sr
            if not text or len(text) < 5 or len(text) > 200:
                skipped += 1
                continue
            if duration < 1.0 or duration > 10.0:
                skipped += 1
                continue
            
            # Normalize
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / 32768.0
            
            # Save
            audio_path = f"./data/audio/malayalam_{len(processed_data):06d}.wav"
            sf.write(audio_path, audio_array, sr)
            
            processed_data.append({
                'audio_path': audio_path,
                'text': text,
                'sample_rate': int(sr),
                'duration': float(duration),
                'text_length': len(text)
            })
            
        except Exception as e:
            if skipped < 5:
                print(f"\nError at {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\n✓ Processed {len(processed_data)} samples")
    print(f"✗ Skipped {skipped} samples")
    
    if len(processed_data) == 0:
        raise ValueError("No samples processed!")
    
    ds = Dataset.from_list(processed_data)
    train_test = ds.train_test_split(test_size=0.1, seed=42)
    train_test.save_to_disk("./data/processed")
    
    stats = {
        'total_samples': len(processed_data),
        'train_samples': len(train_test['train']),
        'test_samples': len(train_test['test']),
        'skipped': skipped,
        'avg_duration': float(np.mean([x['duration'] for x in processed_data])),
        'avg_text_length': float(np.mean([x['text_length'] for x in processed_data]))
    }
    
    with open('./data/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ SUCCESS!")
    print(f"{'='*60}")
    print(f"Total:  {stats['total_samples']}")
    print(f"Train:  {stats['train_samples']}")
    print(f"Test:   {stats['test_samples']}")
    print(f"Avg duration: {stats['avg_duration']:.2f}s")
    print(f"{'='*60}\n")
    
    return train_test

if __name__ == "__main__":
    prepare_malayalam_tts_data()
