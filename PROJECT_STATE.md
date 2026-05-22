# MiniCPM-AV Project: Current State

**Last Updated:** 2026-05-22  
**Project:** MiniCPM-AV (Audio-Enhanced MiniCPM-V for Audio-Visual Question Answering)

---

## 1. Executive Summary

MiniCPM-AV is an audio-visual question answering model that extends MiniCPM-V with audio understanding capabilities via the Moonshine ASR encoder. The model was successfully trained on the MUSIC-AVQA dataset and is now available on HuggingFace Hub.

---

## 2. Training Completed ✅

### 2.1 Training Configuration
| Parameter | Value |
|-----------|-------|
| **Dataset** | MUSIC-AVQA (78,789 train / 19,938 test samples) |
| **Epochs** | 3 |
| **Batch Size** | 1 |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Precision** | fp16 (mixed) |
| **Gradient Checkpointing** | Enabled |
| **Hardware** | NVIDIA RTX 5090 (32 GB VRAM) |

### 2.2 Training Outcome
| Metric | Value |
|--------|-------|
| **Final Best Validation Loss** | 0.3194 |
| **Trainable Parameters** | ~91.8M (vs 3.5B full model) |
| **Training Time** | ~16.5 hours |
| **Status** | ✅ Completed Successfully |

### 2.3 Checkpoints Saved
| Checkpoint | Location | Size | Notes |
|------------|----------|------|-------|
| `best_model` | `/checkpoints/minicpm-av/best_model/` | 258.1 MB | Lowest val_loss: 0.3194 |
| `checkpoint_epoch_1` | `/checkpoints/minicpm-av/checkpoint_epoch_1/` | 258.1 MB | Epoch 1 |
| `checkpoint_epoch_2` | `/checkpoints/minicpm-av/checkpoint_epoch_2/` | 258.1 MB | Epoch 2 |
| `checkpoint_epoch_3` | `/checkpoints/minicpm-av/checkpoint_epoch_3/` | 258.1 MB | Final |

---

## 3. Model Architecture

### 3.1 Component Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniCPM-AV Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Input ──► Moonshine Encoder ──► Audio Projector ──┐      │
│  (raw audio)     (UsefulSensors/      (288 → 2304)        │      │
│                  moonshine-tiny)     + Token Compressor  │      │
│                                                          │      │
│  Image Input ──► SigLIP Vision ───────────────────────┐  │      │
│  (image)         Encoder                              │  │      │
│                                                       ▼  ▼      │
│                                            ┌─────────────────┐  │
│                                            │  Concatenated   │  │
│                                            │  Audio + Vision │  │
│                                            │  + Text Tokens  │  │
│                                            └────────┬────────┘  │
│                                                   │           │
│                                                   ▼           │
│                                            ┌─────────────────┐  │
│                                            │ MiniCPM-V LLM   │  │
│                                            │ (with LoRA)     │  │
│                                            └────────┬────────┘  │
│                                                   │           │
│                                                   ▼           │
│                                            Answer Output      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Modular Components

| Component | Source | Trainable | Size | Purpose |
|-----------|--------|-----------|------|---------|
| **MiniCPM-V** | `openbmb/MiniCPM-V` | ❌ Frozen (base) | ~3.5B | Vision-language backbone |
| **Moonshine** | `UsefulSensors/moonshine-tiny` | ❌ Frozen | ~10M | Audio encoder |
| **Audio Projector** | Custom | ✅ Yes | ~1.3M | 288 → 2304 projection |
| **Token Compressor** | Custom | ✅ Yes | ~115K | 50-token compression |
| **Modality Embeddings** | Custom | ✅ Yes | ~7K | Audio/vision/text distinction |
| **LoRA Adapters** | PEFT | ✅ Yes | ~90M | Efficient LLM fine-tuning |

### 3.3 Checkpoint Contents (`audio_components.pt`)

The 258 MB checkpoint contains ONLY the trainable components:

```python
{
    'audio_projector': AudioProjector.state_dict(),      # ~1.3M params
    'audio_compressor': AudioTokenCompressor.state_dict(), # ~115K params
    'modality_embeddings': ModalityEmbeddings.state_dict(), # ~7K params
    # LoRA weights are embedded in the checkpoint via PEFT
}
```

**Why this design?**
- ✅ Efficient: 258 MB vs 3.5+ GB full model
- ✅ Flexible: Update base models independently
- ✅ Shareable: Easy distribution and versioning
- ✅ Modular: Audio components are plug-and-play

---

## 4. HuggingFace Hub Upload ✅

### 4.1 Repository Details
| Property | Value |
|----------|-------|
| **Repository** | `gvij/minicpm-av-music-avqa` |
| **URL** | https://huggingface.co/gvij/minicpm-av-music-avqa |
| **Visibility** | Public |
| **License** | Apache 2.0 |

### 4.2 Uploaded Files

**Checkpoints:**
- ✅ `best_model/audio_components.pt` (258.1 MB)
- ✅ `checkpoint_epoch_1/audio_components.pt` (258.1 MB)
- ✅ `checkpoint_epoch_2/audio_components.pt` (258.1 MB)
- ✅ `checkpoint_epoch_3/audio_components.pt` (258.1 MB)

**Source Code:**
- ✅ `src/modeling_minicpm_av.py`
- ✅ `src/audio_encoder.py`
- ✅ `src/audio_projector.py`
- ✅ `src/data_loader.py`
- ✅ `src/train.py`
- ✅ `src/eval.py`

**Logs & Documentation:**
- ✅ `logs/training_full.log` (50 MB)
- ✅ `logs/training.log`
- ✅ `README.md` (Model card with usage examples)

---

## 5. How to Use

### 5.1 Installation

```bash
# Clone the repository (if needed)
git clone https://huggingface.co/gvij/minicpm-av-music-avqa

# Install dependencies
pip install torch transformers peft huggingface_hub
```

### 5.2 Loading the Model

```python
from modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig
from huggingface_hub import hf_hub_download
import torch

# Initialize model (downloads base MiniCPM-V + Moonshine)
config = MiniCPMAVConfig()
model = MiniCPMAV(config)
model.eval()

# Load trained audio components from HuggingFace
checkpoint_path = hf_hub_download(
    repo_id='gvij/minicpm-av-music-avqa',
    filename='best_model/audio_components.pt'
)
model.load_pretrained(checkpoint_path)

print("Model loaded successfully!")
```

### 5.3 Inference Example

```python
import torch
from PIL import Image

# Prepare inputs
audio = torch.randn(1, 16000)  # 1 second of 16kHz audio
image = Image.open("path/to/image.jpg")
question = "What instrument is playing in the video?"

# Generate answer
with torch.no_grad():
    answer = model.generate(
        audio=audio,
        image=image,
        question=question,
        max_new_tokens=100
    )

print(f"Answer: {answer}")
```

### 5.4 Using Different Checkpoints

```python
# Use epoch 3 checkpoint (final)
checkpoint_path = hf_hub_download(
    repo_id='gvij/minicpm-av-music-avqa',
    filename='checkpoint_epoch_3/audio_components.pt'
)

# Or use a specific epoch
checkpoint_path = hf_hub_download(
    repo_id='gvij/minicpm-av-music-avqa',
    filename='checkpoint_epoch_1/audio_components.pt'
)
```

---

## 6. Local File Structure

```
/root/minicpm-av/
├── README.md                          # Project overview
├── PROJECT_STATE.md                   # This file
├── TRAINING_REPORT.md                 # Detailed training report
├── TRAINING_STEPS.md                  # Training procedure docs
├── requirements.txt                   # Python dependencies
├── upload_to_hf.py                   # HF upload script
├── edge_deployment_report.md          # Edge deployment analysis
│
├── src/                              # Source code
│   ├── modeling_minicpm_av.py        # Main model class
│   ├── audio_encoder.py              # Moonshine wrapper
│   ├── audio_projector.py            # Projection + compressor
│   ├── data_loader.py                # MUSIC-AVQA data loading
│   ├── train.py                      # Training script
│   ├── eval.py                       # Evaluation script
│   ├── edge_profile.py               # Edge profiling tools
│   └── verify_models.py              # Model verification
│
├── checkpoints/                      # Training checkpoints
│   └── minicpm-av/
│       ├── best_model/
│       │   └── audio_components.pt   # Best checkpoint
│       ├── checkpoint_epoch_1/
│       │   └── audio_components.pt
│       ├── checkpoint_epoch_2/
│       │   └── audio_components.pt
│       ├── checkpoint_epoch_3/
│       │   └── audio_components.pt
│       ├── logs/                     # TensorBoard logs
│       ├── training.log              # Training log
│       └── training_full.log         # Detailed log
│
├── data/                             # Dataset cache
│   └── cache/
│       └── DraculaDragon___music-avqa-v2.0/
│
├── plans/                            # Planning documents
│   ├── plan_audio_integration.md
│   └── plan_mtp_research.md
│
└── venv/                             # Python virtual environment
```

---

## 7. What Was Fixed During Training

### 7.1 Critical Bug: LoRA Not Applied
**Problem:** Original script had 3.1B trainable parameters causing OOM errors.

**Solution:**
- Properly apply LoRA to `model.minicpm.llm` using PEFT's `get_peft_model()`
- Freeze all parameters except audio components + LoRA adapters
- Add gradient checkpointing and mixed precision support

**Result:** Reduced trainable params from 3.1B to ~91.8M

### 7.2 Upload Script Bug
**Problem:** String formatting error in `create_model_card()` function.

**Solution:** Fixed `.format()` call on triple-quoted string with curly braces.

---

## 8. What's Left / Next Steps

### 8.1 Evaluation & Validation ⏳
- [ ] Run full evaluation on MUSIC-AVQA test set
- [ ] Compute accuracy metrics (exact match, F1, etc.)
- [ ] Generate confusion matrix
- [ ] Compare against baseline (MiniCPM-V without audio)
- [ ] Qualitative analysis: sample predictions

### 8.2 Inference Optimization ⏳
- [ ] Quantize model (INT8/INT4) for faster inference
- [ ] Implement batch inference
- [ ] Add streaming audio support
- [ ] Optimize for real-time applications

### 8.3 Edge Deployment ⏳
- [ ] Convert to ONNX format
- [ ] Test on edge devices (Jetson, mobile)
- [ ] Profile memory and latency
- [ ] Create deployment package

### 8.4 Documentation ⏳
- [ ] Add inference examples to README
- [ ] Create Colab demo notebook
- [ ] Write API documentation
- [ ] Add troubleshooting guide

### 8.5 Future Enhancements (Optional)
- [ ] Fine-tune on other AVQA datasets
- [ ] Experiment with different audio encoders
- [ ] Add video temporal understanding
- [ ] Multi-language support

---

## 9. Key Metrics Summary

| Metric | Value |
|--------|-------|
| **Training Samples** | 78,789 |
| **Test Samples** | 19,938 |
| **Training Epochs** | 3 |
| **Best Val Loss** | 0.3194 |
| **Trainable Params** | ~91.8M |
| **Checkpoint Size** | 258.1 MB |
| **Training Time** | ~16.5 hours |
| **Hardware Used** | RTX 5090 (32 GB) |

---

## 10. References

- **Base Model:** [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V) by OpenBMB
- **Audio Encoder:** [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) by Useful Sensors
- **Dataset:** [MUSIC-AVQA](https://huggingface.co/datasets/DraculaDragon/music-avqa-v2.0)
- **HuggingFace Repo:** https://huggingface.co/gvij/minicpm-av-music-avqa

---

## 11. Contact & Support

For issues or questions:
1. Check the HuggingFace model card: https://huggingface.co/gvij/minicpm-av-music-avqa
2. Review training logs in `/checkpoints/minicpm-av/training_full.log`
3. Refer to `TRAINING_REPORT.md` for detailed training analysis

---

*Document generated: 2026-05-22*  
*Project status: Training complete ✅ | Upload complete ✅ | Ready for evaluation ⏳*
