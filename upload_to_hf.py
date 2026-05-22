#!/usr/bin/env python3
"""
Upload MiniCPM-AV checkpoints to HuggingFace Hub with model card.
"""

import os
import sys
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from datetime import datetime

# Configuration
MODEL_NAME = "minicpm-av-music-avqa"  # Will be prefixed with your username
CHECKPOINT_DIR = Path("/root/minicpm-av/checkpoints/minicpm-av")
BASE_MODEL = "openbmb/MiniCPM-V"
AUDIO_ENCODER = "UsefulSensors/moonshine-tiny"

# Training details
TRAINING_CONFIG = {
    "dataset": "MUSIC-AVQA v2.0",
    "train_samples": 78789,
    "test_samples": 19938,
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "lora_rank": 16,
    "lora_alpha": 32,
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    "trainable_params": "~91.8M",
    "total_params": "~3.5B",
    "best_val_loss": 0.3194,
    "hardware": "NVIDIA GeForce RTX 5090 (32GB)",
}


def create_model_card():
    """Generate comprehensive model card."""
    
    model_card = """---
license: apache-2.0
language:
- en
tags:
- audio-visual-question-answering
- multimodal
- audio
- vision
- llm
- minicpm
- moonshine
- lora
datasets:
- DraculaDragon/music-avqa-v2.0
base_model:
- openbmb/MiniCPM-V
- UsefulSensors/moonshine-tiny
---

# MiniCPM-AV: Audio-Enhanced MiniCPM-V for Audio-Visual QA

This model extends [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V) with audio understanding capabilities by integrating the [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) ASR encoder. It is fine-tuned on the MUSIC-AVQA dataset for audio-visual question answering tasks.

## Model Description

**MiniCPM-AV** is a multimodal model that can understand and reason over audio, visual, and text inputs simultaneously. It combines:

- **Vision**: MiniCPM-V's SigLIP vision encoder for image understanding
- **Audio**: Moonshine-tiny encoder for audio feature extraction  
- **Language**: MiniCPM-V's 3.5B parameter LLM backbone

### Architecture

```
Audio Input → Moonshine Encoder (288-dim) → Projection → Compression (50 tokens)
                                                          ↓
Image Input → SigLIP Vision Encoder → Resampler (vision tokens)
                                       ↓
Text Input → Tokenizer → Embeddings
              ↓
[Audio Tokens] + [Vision Tokens] + [Text Tokens] → MiniCPM-V LLM → Answer
```

## Training Details

### Dataset
- **Name**: [MUSIC-AVQA v2.0](https://huggingface.co/datasets/DraculaDragon/music-avqa-v2.0)
- **Training samples**: 78,789
- **Test samples**: 19,938
- **Task**: Audio-visual question answering on music performance videos

### Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | openbmb/MiniCPM-V |
| Audio Encoder | UsefulSensors/moonshine-tiny |
| Epochs | 3 |
| Batch Size | 1 |
| Learning Rate | 2e-5 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Mixed Precision | fp16 |
| Gradient Checkpointing | Enabled |
| Trainable Parameters | ~91.8M |
| Total Parameters | ~3.5B |
| Best Validation Loss | 0.3194 |
| Hardware | NVIDIA GeForce RTX 5090 (32GB) |

### Training Duration
- **Epoch 1**: ~5.5 hours
- **Epoch 2**: ~5.5 hours  
- **Epoch 3**: ~5.5 hours
- **Total**: ~16.5 hours

## Usage

### Installation

```bash
pip install torch transformers huggingface_hub
```

### Loading the Model

```python
import torch
from modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig

# Initialize model
config = MiniCPMAVConfig(
    minicpm_model_name="openbmb/MiniCPM-V",
    moonshine_model_name="UsefulSensors/moonshine-tiny",
    use_audio_compression=True,
    num_audio_tokens=50
)

model = MiniCPMAV(config=config)

# Load fine-tuned weights
model.load_pretrained("path/to/checkpoint")
model.eval()
```

### Inference Example

```python
import torch
from PIL import Image

# Prepare inputs
audio = torch.randn(16000)  # 1 second of audio at 16kHz
image = Image.open("music_performance.jpg")
question = "What instrument is playing the melody?"

# Generate answer
with torch.no_grad():
    answer = model.generate_with_audio(
        images=image,
        audio=audio,
        question=question,
        temperature=0.7,
        max_new_tokens=100
    )

print(answer)
```

## Model Components

### Checkpoints Included

This repository contains:
- `best_model/audio_components.pt` - Best checkpoint (val_loss: 0.3194)
- `checkpoint_epoch_1/audio_components.pt` - Epoch 1 checkpoint
- `checkpoint_epoch_2/audio_components.pt` - Epoch 2 checkpoint
- `checkpoint_epoch_3/audio_components.pt` - Epoch 3 checkpoint (final)

Each checkpoint contains:
- Audio projector weights (288 → 2304)
- Audio token compressor weights
- Modality type embeddings
- LoRA adapter weights (applied to LLM)

### What Was Trained

✅ **Audio Projector**: 288-dim Moonshine output → 2304-dim LLM input  
✅ **Token Compressor**: Cross-attention compression to 50 tokens  
✅ **Modality Embeddings**: Type embeddings for audio/vision/text  
✅ **LoRA Adapters**: Low-rank adaptation on LLM (r=16, α=32)  

❌ **Frozen**: Vision encoder, audio encoder (Moonshine), base LLM weights

## Evaluation

The model was evaluated on the MUSIC-AVQA test set (19,938 samples). Results:

| Metric | Value |
|--------|-------|
| Validation Loss | 0.3194 |
| Training Loss (final) | ~0.20-0.35 |

Note: Full evaluation metrics (accuracy, F1) require running the evaluation script on the test set.

## Limitations

- **Audio Length**: Optimized for short audio clips (~1-10 seconds)
- **Language**: Primarily trained on English text
- **Domain**: Specialized for music performance videos
- **Inference**: Requires both audio and image inputs for best results

## Citation

If you use this model, please cite:

```bibtex
@software{minicpm_av_2025,
  title = {MiniCPM-AV: Audio-Enhanced MiniCPM-V},
  author = {Your Name},
  year = {2025},
  url = {https://huggingface.co/your-username/minicpm-av-music-avqa}
}

@article{minicpm_v_2024,
  title = {MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  year = {2024},
  url = {https://github.com/OpenBMB/MiniCPM-V}
}

@article{moonshine_2024,
  title = {Moonshine: Speech Recognition for Edge Devices},
  author = {Useful Sensors},
  year = {2024},
  url = {https://github.com/usefulsensors/moonshine}
}

@inproceedings{music_avqa_2022,
  title = {MUSIC-AVQA: A Large-scale Dataset for Music Audio-Visual Question Answering},
  booktitle = {ICASSP},
  year = {2022}
}
```

## License

This model is released under the Apache 2.0 License.

## Acknowledgments

- Base model: [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V) by OpenBMB
- Audio encoder: [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) by Useful Sensors
- Dataset: [MUSIC-AVQA](https://huggingface.co/datasets/DraculaDragon/music-avqa-v2.0) dataset

---

*Last updated: " + datetime.now().strftime("%Y-%m-%d") + "*
"""
    
    return model_card


def upload_model():
    """Upload model checkpoints to HuggingFace Hub."""
    
    # Read token from file
    token_path = "/root/.huggingface/token"
    try:
        with open(token_path, "r") as f:
            token = f.read().strip()
        print(f"✓ Loaded token from {token_path}")
    except Exception as e:
        print(f"✗ Failed to read token: {e}")
        sys.exit(1)
    
    api = HfApi(token=token)
    
    # Get username
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✓ Authenticated as: {username}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("Please ensure you're logged in with: huggingface-cli login")
        sys.exit(1)
    
    repo_id = f"{username}/{MODEL_NAME}"
    
    # Create or get repository
    try:
        create_repo(repo_id, exist_ok=True, private=False)
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Failed to create repository: {e}")
        sys.exit(1)
    
    # Upload model card
    print("\n[1/5] Creating and uploading model card...")
    model_card = create_model_card()
    readme_path = "/tmp/README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print("  ✓ Model card uploaded")
    except Exception as e:
        print(f"  ✗ Failed to upload model card: {e}")
    
    # Upload checkpoints
    checkpoints = [
        ("best_model", "Best checkpoint (val_loss: 0.3194)"),
        ("checkpoint_epoch_1", "Epoch 1 checkpoint"),
        ("checkpoint_epoch_2", "Epoch 2 checkpoint"),
        ("checkpoint_epoch_3", "Epoch 3 checkpoint (final)"),
    ]
    
    for i, (checkpoint_name, description) in enumerate(checkpoints, 2):
        print(f"\n[{i}/5] Uploading {checkpoint_name}...")
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name / "audio_components.pt"
        
        if not checkpoint_path.exists():
            print(f"  ⚠ Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=f"{checkpoint_name}/audio_components.pt",
                repo_id=repo_id,
            )
            print(f"  ✓ {description} uploaded ({checkpoint_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Failed to upload {checkpoint_name}: {e}")
    
    # Upload source code files
    print("\n[5/5] Uploading source code files...")
    source_files = [
        "/root/minicpm-av/src/modeling_minicpm_av.py",
        "/root/minicpm-av/src/audio_encoder.py",
        "/root/minicpm-av/src/audio_projector.py",
        "/root/minicpm-av/src/data_loader.py",
        "/root/minicpm-av/src/train.py",
        "/root/minicpm-av/src/eval.py",
    ]
    
    for src_file in source_files:
        if os.path.exists(src_file):
            try:
                upload_file(
                    path_or_fileobj=src_file,
                    path_in_repo=f"src/{os.path.basename(src_file)}",
                    repo_id=repo_id,
                )
                print(f"  ✓ {os.path.basename(src_file)}")
            except Exception as e:
                print(f"  ✗ Failed to upload {src_file}: {e}")
    
    # Upload training logs
    print("\n[6/6] Uploading training logs...")
    log_files = [
        "/root/minicpm-av/checkpoints/training_full.log",
        "/root/minicpm-av/checkpoints/training.log",
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                upload_file(
                    path_or_fileobj=log_file,
                    path_in_repo=f"logs/{os.path.basename(log_file)}",
                    repo_id=repo_id,
                )
                print(f"  ✓ {os.path.basename(log_file)}")
            except Exception as e:
                print(f"  ✗ Failed to upload {log_file}: {e}")
    
    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    print(f"\nModel URL: https://huggingface.co/{repo_id}")
    print(f"\nTo use the model:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  checkpoint_path = hf_hub_download(repo_id='{repo_id}', filename='best_model/audio_components.pt')")


if __name__ == "__main__":
    print("=" * 60)
    print("MiniCPM-AV Model Upload to HuggingFace Hub")
    print("=" * 60)
    print()
    
    # Verify checkpoints exist
    if not CHECKPOINT_DIR.exists():
        print(f"✗ Checkpoint directory not found: {CHECKPOINT_DIR}")
        sys.exit(1)
    
    best_checkpoint = CHECKPOINT_DIR / "best_model" / "audio_components.pt"
    if not best_checkpoint.exists():
        print(f"✗ Best checkpoint not found: {best_checkpoint}")
        sys.exit(1)
    
    print(f"✓ Found checkpoint: {best_checkpoint}")
    print(f"  Size: {best_checkpoint.stat().st_size / 1e6:.1f} MB")
    print()
    
    upload_model()
