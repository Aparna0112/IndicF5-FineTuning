from huggingface_hub import HfApi, hf_hub_download
import os

api = HfApi()

print("Checking hareeshbabu82/TeluguIndicF5 repository...")
print("="*60)

try:
    # List all files in the repo
    files = api.list_repo_files(repo_id="hareeshbabu82/TeluguIndicF5")
    
    print("\nFiles in repository:")
    for file in files:
        print(f"  - {file}")
    
    # Get model info
    model_info = api.model_info("hareeshbabu82/TeluguIndicF5")
    
    print("\n" + "="*60)
    print("Model Info:")
    print(f"Model ID: {model_info.modelId}")
    print(f"Downloads: {model_info.downloads}")
    print(f"Likes: {model_info.likes}")
    
    if model_info.card_data:
        print(f"\nModel Card Data: {model_info.card_data}")
    
    # Try to download and check config
    try:
        config_path = hf_hub_download(
            repo_id="hareeshbabu82/TeluguIndicF5",
            filename="config.json"
        )
        print(f"\nConfig downloaded to: {config_path}")
        
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
            print("\nConfig contents:")
            print(json.dumps(config, indent=2))
    except:
        print("\nNo config.json found")
    
    # Check for training script or README
    try:
        readme_path = hf_hub_download(
            repo_id="hareeshbabu82/TeluguIndicF5",
            filename="README.md"
        )
        print(f"\nREADME found at: {readme_path}")
        with open(readme_path, 'r') as f:
            readme = f.read()
            print("\nREADME contents:")
            print(readme[:1000])  # First 1000 chars
    except:
        print("\nNo README.md found")
        
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "="*60)
print("✓ Check complete")
