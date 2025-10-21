# train_indicf5.py - Train IndicF5 for Malayalam
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

from f5_tts.train.finetune_cli import main as finetune_main
import sys

# Set config path
config_path = "./configs/malayalam_train.yaml"

print("="*60)
print("Malayalam IndicF5 Fine-tuning")
print("="*60)

# Run fine-tuning
sys.argv = [
    'train_indicf5.py',
    '--config', config_path,
    '--model', 'F5TTS_Base',
    '--ckpt_path', '',  # Start from scratch or provide pretrained path
]

finetune_main()
