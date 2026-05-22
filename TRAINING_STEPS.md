# MiniCPM-AV Training Guide

Complete step-by-step instructions to set up the environment, download data, and train the MiniCPM-AV model.

---

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with ≥24GB VRAM (A100 40GB/80GB recommended, RTX 4090 works)
- **Storage**: ~50GB free space (models + dataset)
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10+

---

## Step 1: Clone Repository

```bash
git clone https://github.com/gauravvij/minicpm-av.git
cd minicpm-av
```

---

## Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

---

## Step 3: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0
pip install datasets accelerate peft tensorboard
pip install sentencepiece protobuf
pip install timm
pip install pillow  # For image processing
pip install soundfile librosa  # For audio processing (optional, for custom data)
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 4: Download Models (Automatic on First Run)

Models are downloaded automatically when you first run training, but you can pre-download:

```bash
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading MiniCPM-V...')
AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
print('Downloading Moonshine...')
AutoModel.from_pretrained('UsefulSensors/moonshine-tiny')
print('Done!')
"
```

Models are cached to `~/.cache/huggingface/hub/` (~15GB total).

---

## Step 5: Download Dataset

The dataset is loaded automatically from HuggingFace, but you can verify access:

```bash
python -c "
from datasets import load_dataset
print('Loading MUSIC-AVQA-v2.0...')
dataset = load_dataset('DraculaDragon/MUSIC-AVQA-v2.0')
print(f'Train samples: {len(dataset[\"train\"])}')
print(f'Test samples: {len(dataset[\"test\"])}')
print('Dataset ready!')
"
```

---

## Step 6: Verify Setup

Run the model test to ensure everything works:

```bash
cd src
python modeling_minicpm_av.py
```

Expected output:
```
Testing MiniCPM-AV Model
...
✓ All MiniCPM-AV tests passed!
```

---

## Step 7: Start Training

### Quick Test (10 samples, 1 epoch)
```bash
python src/train.py \
    --max_train_samples 10 \
    --num_epochs 1 \
    --batch_size 1 \
    --output_dir ./checkpoints/test
```

### Full Training (Recommended)
```bash
python src/train.py \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --save_steps 1000 \
    --eval_steps 500 \
    --logging_steps 100 \
    --output_dir ./checkpoints/minicpm-av \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_workers 4
```

### Training with Gradient Accumulation (for smaller GPUs)
```bash
python src/train.py \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --output_dir ./checkpoints/minicpm-av \
    --use_lora
```

---

## Step 8: Monitor Training

### TensorBoard
```bash
tensorboard --logdir ./checkpoints/minicpm-av/logs
```
Open http://localhost:6006 in browser.

### Command Line
Watch training logs:
```bash
tail -f checkpoints/minicpm-av/training.log
```

---

## Step 9: Verify Checkpoint Contents (IMPORTANT)

Before evaluating, verify that your checkpoint contains LoRA weights:

```bash
# Check checkpoint structure
ls -la ./checkpoints/minicpm-av/best_model/

# Expected output:
# audio_components.pt      # Audio projector, compressor, embeddings
# lora_adapters/           # LoRA weights (directory)
# └── adapter_config.json
# └── adapter_model.safetensors (or .bin)
```

**⚠️ Critical:** If `lora_adapters/` folder is missing, the checkpoint does NOT contain the trained QA capability. This happens if using an older version of `modeling_minicpm_av.py`. Retrain with the fixed code.

**Verify LoRA weights are saved:**
```bash
python -c "
import os
import torch

checkpoint_dir = './checkpoints/minicpm-av/best_model'
audio_file = os.path.join(checkpoint_dir, 'audio_components.pt')
lora_dir = os.path.join(checkpoint_dir, 'lora_adapters')

# Check audio components
if os.path.exists(audio_file):
    ckpt = torch.load(audio_file, map_location='cpu')
    print(f'✓ Audio components: {list(ckpt.keys())}')
else:
    print('✗ Missing audio_components.pt')

# Check LoRA adapters
if os.path.exists(lora_dir):
    print(f'✓ LoRA adapters directory exists')
    lora_files = os.listdir(lora_dir)
    print(f'  Files: {lora_files}')
else:
    print('✗ MISSING: lora_adapters/ directory - checkpoint incomplete!')
"
```

## Step 10: Evaluate Trained Model

```bash
python src/eval.py \
    --checkpoint_path ./checkpoints/minicpm-av/best_model \
    --output_dir ./results
```

This runs:
- AVQA accuracy on test set
- Modality ablation (audio-only, vision-only, audio+vision)

**Note:** If evaluation shows 0-2% accuracy, check that LoRA weights are present in the checkpoint (see Step 9).

---

## Step 10: Inference Example

```python
from src.modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig
import torch

# Load model
config = MiniCPMAVConfig()
model = MiniCPMAV(config)
model.load_pretrained('./checkpoints/minicpm-av/best_model')
model.eval()

# Prepare inputs
audio = torch.randn(16000)  # 1 second of audio
image = PIL.Image.open('image.jpg')
question = "What instrument is playing?"

# Generate answer
answer = model.generate_with_audio(
    images=image,
    audio=audio,
    question=question
)
print(answer)
```

---

## 📊 Expected Training Metrics

| Metric | Expected Value |
|--------|----------------|
| Training Loss (start) | ~2.5-3.5 |
| Training Loss (end) | ~0.5-1.5 |
| AVQA Accuracy | 60-75% (depends on training) |
| Training Time (A100) | 6-8 hours for 3 epochs |
| Training Time (RTX 4090) | 12-16 hours for 3 epochs |

---

## 🔧 Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 1 or 2
- Increase `--gradient_accumulation_steps`
- Enable `--gradient_checkpointing` (if implemented)

### Slow Training
- Increase `--num_workers` for data loading
- Ensure dataset is cached locally
- Use SSD for data storage

### Model Not Loading
- Check `transformers==4.49.0` is installed
- Verify `trust_remote_code=True` is set
- Check internet connection for model download

### Dataset Download Fails
- Check HuggingFace access: `huggingface-cli login`
- Dataset is public, should not need token
- Try manual download from https://huggingface.co/datasets/DraculaDragon/MUSIC-AVQA-v2.0

---

## 📁 Directory Structure After Training

```
minicpm-av/
├── src/
│   ├── audio_encoder.py
│   ├── audio_projector.py
│   ├── modeling_minicpm_av.py
│   ├── data_loader.py
│   ├── train.py
│   └── eval.py
├── checkpoints/
│   └── minicpm-av/
│       ├── best_model/
│       │   └── audio_components.pt
│       ├── checkpoint_epoch_1/
│       ├── checkpoint_epoch_2/
│       ├── checkpoint_epoch_3/
│       └── logs/
├── results/
│   └── eval_results.json
└── data/
    └── cache/  # HuggingFace cache
```

---

## 🚀 Next Steps

1. **Export to ONNX**: Convert for edge deployment
2. **Quantize**: Use GGUF/AWQ for smaller size
3. **Fine-tune**: On your own AVQA dataset
4. **Evaluate**: Run full modality ablation study

---

## 📚 References

- MiniCPM-V: https://github.com/OpenBMB/MiniCPM-V
- Moonshine ASR: https://github.com/usefulsensors/moonshine
- MUSIC-AVQA Dataset: https://github.com/YapengTian/MUSIC-AVQA

---

**Last Updated**: 2025-01-19  
**Contact**: Open an issue on GitHub for questions
