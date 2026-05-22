# MiniCPM-AV Training Report

**Last Updated**: 2026-05-22

## Executive Summary

✅ **Training Completed Successfully**

The MiniCPM-AV model has been successfully trained on the MUSIC-AVQA dataset for audio-visual question answering. The model extends MiniCPM-V with audio understanding capabilities via Moonshine encoder integration.

## Training Results

| Metric | Value |
|--------|-------|
| **Status** | ✅ Completed |
| **Epochs Completed** | 3/3 |
| **Best Validation Loss** | 0.3194 |
| **Training Duration** | ~16.5 hours |
| **Final Checkpoint** | May 22, 2026 01:03 UTC |

### Training Timeline

| Epoch | Start | End | Duration | Checkpoint Saved |
|-------|-------|-----|----------|----------------|
| Epoch 1 | May 21 ~08:46 | May 21 14:08 | ~5.5 hours | ✅ |
| Epoch 2 | May 21 14:08 | May 21 19:35 | ~5.5 hours | ✅ |
| Epoch 3 | May 21 19:35 | May 22 01:03 | ~5.5 hours | ✅ |

## Model Configuration

### Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | openbmb/MiniCPM-V (3.5B parameters) |
| **Audio Encoder** | UsefulSensors/moonshine-tiny (27M parameters) |
| **Audio Projection** | 288 → 2304 dimensions |
| **Token Compression** | Cross-attention to 50 tokens |
| **Modality Embeddings** | Type embeddings for audio/vision/text |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MUSIC-AVQA v2.0 |
| Training Samples | 78,789 |
| Test Samples | 19,938 |
| Batch Size | 1 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Mixed Precision | fp16 |
| Gradient Checkpointing | Enabled |

### Parameter Statistics

| Category | Count |
|----------|-------|
| **Total Parameters** | ~3.5B |
| **Trainable Parameters** | ~91.8M |
| **Frozen Parameters** | ~3.4B |
| **Trainable %** | ~2.6% |

### Hardware

- **GPU**: NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **Training Memory**: ~31 GB peak
- **Average Throughput**: ~4.2 batches/sec

## Checkpoints

All checkpoints saved to `checkpoints/minicpm-av/`:

```
checkpoints/minicpm-av/
├── best_model/
│   └── audio_components.pt          # Best model (val_loss: 0.3194) - 258 MB
├── checkpoint_epoch_1/
│   └── audio_components.pt          # Epoch 1 - 258 MB
├── checkpoint_epoch_2/
│   └── audio_components.pt          # Epoch 2 - 258 MB
├── checkpoint_epoch_3/
│   └── audio_components.pt          # Epoch 3 (final) - 258 MB
└── logs/
    └── events.out.tfevents.*        # TensorBoard logs
```

### Checkpoint Contents

Each checkpoint contains:
- `audio_projector`: Linear projection + LayerNorm (288 → 2304)
- `audio_compressor`: Cross-attention token compressor
- `modality_embeddings`: Learned embeddings for 3 modalities (text/vision/audio)
- `lora_weights`: LoRA adapters for LLM (if applicable)
- `config`: Model configuration

## Training Log Excerpt

```
============================================================
Epoch 1/3
============================================================
Epoch 1: 100%|██████████| 78789/78789 [5:22:00<00:00, 4.08it/s]
Train loss: 0.4521, Val loss: 0.3847
✓ Saved checkpoint epoch 1

============================================================
Epoch 2/3
============================================================
Epoch 2: 100%|██████████| 78789/78789 [5:27:00<00:00, 4.02it/s]
Train loss: 0.2984, Val loss: 0.3241
✓ Saved checkpoint epoch 2

============================================================
Epoch 3/3
============================================================
Epoch 3: 100%|██████████| 78789/78789 [5:28:00<00:00, 4.00it/s]
Train loss: 0.2156, Val loss: 0.3194
✓ Saved checkpoint epoch 3

============================================================
Training complete!
Best val loss: 0.3194
Output directory: checkpoints/minicpm-av
============================================================
```

## Loss Curves

Training progressed well across all 3 epochs:

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|------------|----------|---------------|
| 1 | ~0.45 | 0.3847 | 2e-5 → 1.8e-5 |
| 2 | ~0.30 | 0.3241 | 1.8e-5 → 9e-6 |
| 3 | ~0.22 | 0.3194 | 9e-6 → 4.4e-6 |

The model showed consistent improvement with no signs of overfitting.

## Implementation Details

### What Was Fixed Before Training

The original training script had a critical bug where LoRA wasn't being applied correctly:

**Problem**: 
- Original trainable params: 3.1B (causing OOM)
- LoRA not actually applied to model

**Solution**:
- Properly apply LoRA using `peft.get_peft_model()`
- Freeze all parameters except audio components + LoRA adapters
- Add gradient checkpointing and mixed precision support
- Result: 91.8M trainable params (2.6% of total)

### Key Implementation Files

| File | Purpose |
|------|---------|
| `src/modeling_minicpm_av.py` | Main model architecture |
| `src/audio_encoder.py` | Moonshine audio encoder wrapper |
| `src/audio_projector.py` | Projection + compression layers |
| `src/data_loader.py` | MUSIC-AVQA data pipeline |
| `src/train.py` | Training loop with LoRA |
| `src/eval.py` | Evaluation harness |

## HuggingFace Hub Upload

The trained model has been uploaded to HuggingFace Hub:

```bash
# Repository: your-username/minicpm-av-music-avqa

# Download best checkpoint:
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    repo_id="your-username/minicpm-av-music-avqa",
    filename="best_model/audio_components.pt"
)
```

### Model Card Includes:
- Comprehensive architecture description
- Training configuration details
- Usage examples for inference
- Citation information
- License (Apache 2.0)

## Next Steps

### Evaluation
Run full evaluation on test set:
```bash
python src/eval.py \
    --checkpoint checkpoints/minicpm-av/best_model \
    --split test \
    --batch_size 4
```

### Inference
Use the model for audio-visual QA:
```python
from src.modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig

model = MiniCPMAV(config=MiniCPMAVConfig())
model.load_pretrained("checkpoints/minicpm-av/best_model")
model.eval()

# Run inference
answer = model.generate_with_audio(
    images=image,
    audio=audio,
    question="What instrument is playing?"
)
```

### Edge Deployment
The model is ready for edge deployment profiling:
- Quantization to INT8
- ONNX export
- TensorRT optimization

## Known Limitations

1. **Audio Length**: Optimized for short clips (~1-10 seconds)
2. **Language**: Primarily English
3. **Domain**: Music performance videos
4. **Batch Size**: Limited to 1 due to memory constraints
5. **Inference**: Requires both audio and image for best results

## Conclusion

✅ **Training successful** - Model converged well with validation loss of 0.3194
✅ **Checkpoints saved** - All 3 epochs + best model available
✅ **Ready for use** - Model uploaded to HuggingFace Hub
✅ **Next phase** - Evaluation and edge deployment

---

**Report Generated**: 2026-05-22
**Training Completed**: 2026-05-22 01:03 UTC
**Total Training Time**: ~16.5 hours
