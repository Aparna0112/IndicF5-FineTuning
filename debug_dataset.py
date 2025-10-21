from datasets import load_dataset
import numpy as np

print("Loading dataset...")
dataset = load_dataset("ceymox/MALAYALAM_STT_COMBINED_1_2", split="train")

print(f"Total samples: {len(dataset)}")
print("\nChecking first sample...")

sample = dataset[0]
print("Keys:", list(sample.keys()))

audio_data = sample['audio']
print("\nAudio type:", type(audio_data))

if isinstance(audio_data, dict):
    print("Audio dict keys:", list(audio_data.keys()))
    
    for key, value in audio_data.items():
        if key == 'array':
            print(f"  {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
        else:
            print(f"  {key}: {type(value)} = {value if not isinstance(value, (list, np.ndarray)) else f'len={len(value)}'}")

print("\nTranscription:", sample.get('transcription', 'NOT FOUND'))
print("\n✓ Done")
