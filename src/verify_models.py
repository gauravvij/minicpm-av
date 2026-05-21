#!/usr/bin/env /usr/bin/python3
"""
Verify model loading for MiniCPM-V 4.6 and Moonshine-tiny
"""
import torch
import sys

print("=" * 60)
print("Model Loading Verification")
print("=" * 60)

# Check PyTorch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")

# Test 1: Load Moonshine-tiny
print("\n" + "-" * 60)
print("Test 1: Loading Moonshine-tiny...")
print("-" * 60)
try:
    from transformers import AutoFeatureExtractor, MoonshineModel
    
    moonshine_model = MoonshineModel.from_pretrained(
        "UsefulSensors/moonshine-tiny",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print(f"✓ Moonshine-tiny loaded successfully")
    print(f"  - Hidden size: {moonshine_model.config.hidden_size}")
    print(f"  - Encoder layers: {moonshine_model.config.encoder_num_hidden_layers}")
    print(f"  - Parameters: {sum(p.numel() for p in moonshine_model.parameters()) / 1e6:.1f}M")
    
    # Test forward pass with dummy audio
    import numpy as np
    dummy_audio = torch.randn(1, 16000)  # 1 second at 16kHz
    with torch.no_grad():
        outputs = moonshine_model(dummy_audio, decoder_input_ids=torch.tensor([[1, 1]]))
        print(f"  - Encoder output shape: {outputs.last_hidden_state.shape}")
    
    moonshine_success = True
except Exception as e:
    print(f"✗ Moonshine-tiny failed: {e}")
    moonshine_success = False

# Test 2: Load MiniCPM-V 4.6
print("\n" + "-" * 60)
print("Test 2: Loading MiniCPM-V 4.6...")
print("-" * 60)
try:
    from transformers import AutoModel, AutoTokenizer
    
    # Note: MiniCPM-V 4.6 uses trust_remote_code
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V",
        trust_remote_code=True
    )
    
    print(f"✓ MiniCPM-V loaded successfully")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Check config for architecture details
    if hasattr(model, 'config'):
        config = model.config
        print(f"  - Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
        print(f"  - Vocab size: {config.vocab_size if hasattr(config, 'vocab_size') else 'N/A'}")
    
    minicpm_success = True
except Exception as e:
    print(f"✗ MiniCPM-V failed: {e}")
    import traceback
    traceback.print_exc()
    minicpm_success = False

# Test 3: Check MUSIC-AVQA dataset
print("\n" + "-" * 60)
print("Test 3: Checking MUSIC-AVQA-v2.0 dataset...")
print("-" * 60)
try:
    from datasets import load_dataset
    
    # Just check if dataset exists (don't download full yet)
    dataset_info = load_dataset("DraculaDragon/MUSIC-AVQA-v2.0", split="train", streaming=True)
    sample = next(iter(dataset_info))
    print(f"✓ MUSIC-AVQA-v2.0 accessible")
    print(f"  - Sample keys: {list(sample.keys())}")
    dataset_success = True
except Exception as e:
    print(f"✗ Dataset check failed: {e}")
    dataset_success = False

# Summary
print("\n" + "=" * 60)
print("Verification Summary")
print("=" * 60)
print(f"Moonshine-tiny: {'✓ PASS' if moonshine_success else '✗ FAIL'}")
print(f"MiniCPM-V 4.6: {'✓ PASS' if minicpm_success else '✗ FAIL'}")
print(f"MUSIC-AVQA-v2.0: {'✓ PASS' if dataset_success else '✗ FAIL'}")

if moonshine_success and minicpm_success and dataset_success:
    print("\n✓ All models verified! Ready to proceed.")
    sys.exit(0)
else:
    print("\n✗ Some verifications failed. Check errors above.")
    sys.exit(1)
