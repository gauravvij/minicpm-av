# Plan: Audio-Enhanced MiniCPM-V (MiniCPM-AV)

## Goal
Extend MiniCPM-V 4.6 with audio understanding capability by integrating Moonshine ASR encoder, creating a trimodal (audio-vision-language) model.

## Research Summary
- **Moonshine ASR**: 27M-61M params, encoder-decoder, RoPE embeddings, Apache 2.0
- **MiniCPM-V 4.6**: SigLIP2-400M vision + Qwen3.5-0.8B LLM, modular architecture
- **Dataset**: MUSIC-AVQA-v2.0 (8.1k QA pairs) available on HuggingFace
- **Feasibility**: ✅ Confirmed - all components compatible and available

## Architecture Design

```
Audio Input (30s waveform)
    ↓
Moonshine-tiny Encoder (27M)
    ↓
Encoder Hidden States: [batch, audio_tokens, 288]
    ↓
Audio Projection Layer: Linear(288 → 1024)
    ↓
Audio Tokens: [batch, ~200, 1024]
    ↓
                    Concatenate with Vision Tokens
    ┌─────────────────────────────────────────────────────┐
    ↓                                                   ↓
Image Input                                       Audio Tokens
    ↓                                                   ↓
SigLIP2-400M Vision Encoder                    [batch, ~200, 1024]
    ↓                                                   ↓
NaViT Vision Tokens                              ↓
[batch, ~500, 1024]                              ↓
    ↓                                              ↓
    └──────────────────┬───────────────────────────┘
                       ↓
            Combined Multimodal Tokens
            [batch, ~700, 1024]
                       ↓
            Qwen3.5-0.8B LLM Backbone
                       ↓
                 Text Output
```

## Subtasks

### Phase 1: Foundation (Days 1-2)

1. **Setup Environment**
   - Install dependencies: transformers, torch, datasets, accelerate
   - Download MiniCPM-V 4.6 and Moonshine-tiny
   - Expected output: Working environment with model loading

2. **Implement Audio Encoder Module**
   - Create `AudioEncoder` class wrapping Moonshine
   - Extract encoder hidden states (not decoder outputs)
   - Expected output: `audio_encoder.py` with test script

3. **Implement Audio Projection**
   - Linear projection: 288 → 1024
   - LayerNorm for stability
   - Expected output: `audio_projector.py`

### Phase 2: Integration (Days 3-4)

4. **Extend MiniCPM-V Architecture**
   - Modify model to accept audio inputs
   - Concatenate audio tokens with vision tokens
   - Handle audio-only, vision-only, and audio-vision cases
   - Expected output: `modeling_minicpm_av.py`

5. **Data Pipeline**
   - Load MUSIC-AVQA-v2.0 from HuggingFace
   - Process audio: waveform → Moonshine features
   - Process images: standard MiniCPM-V preprocessing
   - Expected output: `data_loader.py`

### Phase 3: Training (Days 5-7)

6. **Training Script**
   - Freeze vision encoder and audio encoder
   - Train: audio projector + LLM LoRA (or partial layers)
   - Loss: standard language modeling on QA answers
   - Expected output: `train.py` with config

7. **Training Run**
   - Fine-tune on MUSIC-AVQA-v2.0
   - Save checkpoints
   - Expected output: Trained checkpoint

### Phase 4: Evaluation (Days 8-9)

8. **Evaluation Harness**
   - AVQA accuracy metric
   - Modality ablation: audio-only, vision-only, audio-vision
   - Expected output: `eval.py`

9. **Edge Feasibility Test**
   - Measure inference latency on CPU
   - Memory usage profiling
   - Expected output: Edge deployment report

### Phase 5: Documentation (Day 10)

10. **Documentation & Cleanup**
    - README with usage instructions
    - Example inference scripts
    - Expected output: Complete project documentation

## Deliverables

| Path | Description |
|------|-------------|
| `src/audio_encoder.py` | Moonshine wrapper for feature extraction |
| `src/audio_projector.py` | Audio token projection layer |
| `src/modeling_minicpm_av.py` | Extended MiniCPM-V with audio support |
| `src/data_loader.py` | AVQA data pipeline |
| `src/train.py` | Training script |
| `src/eval.py` | Evaluation script |
| `src/inference.py` | Inference demo |
| `checkpoints/` | Trained model checkpoints |
| `results/` | Evaluation results |
| `README.md` | Documentation |

## Evaluation Criteria
- [ ] Audio encoder successfully extracts features from raw audio
- [ ] Model accepts both audio and image inputs simultaneously
- [ ] Training converges (loss decreases, accuracy improves)
- [ ] AVQA accuracy > random baseline (prove audio helps)
- [ ] Modality ablation shows audio contributes meaningfully
- [ ] Edge inference latency < 5s per sample on CPU

## Technical Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Token explosion (~700 tokens) | Audio token compression/pooling to ~50 tokens |
| Modality collapse | Modality dropout during training (randomly mask audio/vision) |
| Synchronization | Use timestamp-aligned audio-visual pairs from AVQA |
| Memory constraints | Freeze encoders, use LoRA for LLM fine-tuning |

## Notes
- **Primary Goal**: Working audio-vision-language model
- **Novel Contribution**: First audio extension of MiniCPM-V
- **Hardware**: GPU for training, CPU for edge testing
- **License**: All components Apache 2.0 - no conflicts
