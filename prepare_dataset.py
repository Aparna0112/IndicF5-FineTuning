import os
import io
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
    try:
        dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2", split="train")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    
    print("\nProcessing audio samples...")
    processed_data = []
    skipped = 0
    total_samples = min(len(dataset), 3000)
    
    for idx in tqdm(range(total_samples), desc="Processing"):
        try:
            example = dataset[idx]
            audio_data = example['audio']
            
            if isinstance(audio_data, dict) and 'array' in audio_data:
                audio_array = np.array(audio_data['array'])
                sr = audio_data.get('sampling_rate', 16000)
            else:
                skipped += 1
                continue
            
            text = example.get('transcription', '').strip()
            if not text or len(text) < 5 or len(text) > 200:
                skipped += 1
                continue
            
            duration = len(audio_array) / sr
            if duration < 1.0 or duration > 10.0:
                skipped += 1
                continue
            
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / 32768.0
            
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
        raise ValueError("No samples were processed!")
    
    print("\nCreating train/test split...")
    ds = Dataset.from_list(processed_data)
    train_test = ds.train_test_split(test_size=0.1, seed=42)
    
    print("Saving dataset...")
    train_test.save_to_disk("./data/processed")
    
    stats = {
        'total_samples': len(processed_data),
        'train_samples': len(train_test['train']),
        'test_samples': len(train_test['test']),
        'skipped_samples': skipped,
        'avg_duration': float(np.mean([x['duration'] for x in processed_data])),
        'avg_text_length': float(np.mean([x['text_length'] for x in processed_data]))
    }
    
    with open('./data/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"Total samples:  {stats['total_samples']}")
    print(f"Train samples:  {stats['train_samples']}")
    print(f"Test samples:   {stats['test_samples']}")
    print(f"Avg duration:   {stats['avg_duration']:.2f}s")
    print("="*60)
    print("\n✓ Ready to train! Run: python3 train.py\n")
    
    return train_test

if __name__ == "__main__":
    prepare_malayalam_tts_data()
