#!/usr/bin/env /usr/bin/python3
"""
Audio Encoder Module for MiniCPM-AV
Wraps Moonshine ASR model to extract encoder hidden states for audio-visual integration.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers import AutoFeatureExtractor, MoonshineModel


class AudioEncoder(nn.Module):
    """
    Audio encoder wrapper around Moonshine ASR model.
    Extracts encoder hidden states (not decoder outputs) for multimodal fusion.
    
    Args:
        model_name: HuggingFace model name for Moonshine
        freeze_encoder: Whether to freeze the encoder weights
        device: Device to load model on
    """
    
    def __init__(
        self,
        model_name: str = "UsefulSensors/moonshine-tiny",
        freeze_encoder: bool = True,
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = MoonshineModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # Move to device
        self.model.to(self.device)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
        
        # Get model config
        self.hidden_size = self.model.config.hidden_size
        self.encoder_layers = self.model.config.encoder_num_hidden_layers
        
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[AudioEncoder] Frozen {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
    
    def preprocess_audio(
        self,
        audio: Union[torch.Tensor, list],
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """
        Preprocess raw audio waveform for Moonshine.
        
        Args:
            audio: Raw audio waveform (tensor or list of tensors)
            sampling_rate: Audio sampling rate (default 16kHz for Moonshine)
            
        Returns:
            Preprocessed input values tensor
        """
        # Handle single audio input
        if isinstance(audio, torch.Tensor) and audio.dim() == 1:
            audio = [audio]
        elif isinstance(audio, torch.Tensor) and audio.dim() == 2:
            audio = [a for a in audio]
        
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        elif isinstance(audio, list) and isinstance(audio[0], torch.Tensor):
            audio = [a.cpu().numpy() for a in audio]
            
        # Use feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        return inputs.input_values.to(self.device)
    
    def forward(
        self,
        audio: Union[torch.Tensor, list],
        sampling_rate: int = 16000,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through audio encoder.
        
        Args:
            audio: Raw audio waveform(s)
            sampling_rate: Audio sampling rate
            return_dict: If True, return dict with hidden_states and attention_mask
            
        Returns:
            If return_dict=True: dict with 'hidden_states' and 'attention_mask'
            If return_dict=False: tuple of (hidden_states, attention_mask)
        """
        # Preprocess audio
        input_values = self.preprocess_audio(audio, sampling_rate)
        
        # Create dummy decoder input IDs (not used for encoder output)
        batch_size = input_values.shape[0]
        decoder_input_ids = torch.tensor([[1, 1]] * batch_size).to(self.device)
        
        # Forward through model
        with torch.set_grad_enabled(not self.model.training or not self.model.parameters().__next__().requires_grad):
            outputs = self.model(
                input_values,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract encoder hidden states (last layer)
        # Shape: [batch, seq_len, hidden_size]
        encoder_hidden_states = outputs.last_hidden_state
        
        # Create attention mask (all real tokens)
        attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=self.device
        )
        
        if return_dict:
            return {
                "hidden_states": encoder_hidden_states,
                "attention_mask": attention_mask,
                "num_tokens": encoder_hidden_states.shape[1]
            }
        else:
            return encoder_hidden_states, attention_mask
    
    def get_audio_features(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[torch.Tensor] = None,
        sampling_rate: int = 16000
    ) -> dict:
        """
        Convenience method to get audio features from file or array.
        
        Args:
            audio_path: Path to audio file (optional)
            audio_array: Audio waveform array (optional)
            sampling_rate: Audio sampling rate
            
        Returns:
            Dictionary with hidden_states and metadata
        """
        if audio_path is not None:
            import soundfile as sf
            audio_array, sr = sf.read(audio_path)
            if sr != sampling_rate:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sampling_rate)
            audio_array = torch.from_numpy(audio_array).float()
        
        if audio_array is None:
            raise ValueError("Must provide either audio_path or audio_array")
            
        return self.forward(audio_array, sampling_rate=sampling_rate, return_dict=True)
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.hidden_size


def test_audio_encoder():
    """Test script for AudioEncoder."""
    print("=" * 60)
    print("Testing AudioEncoder Module")
    print("=" * 60)
    
    # Initialize encoder
    print("\n[1] Initializing AudioEncoder...")
    encoder = AudioEncoder(model_name="UsefulSensors/moonshine-tiny", freeze_encoder=True)
    print(f"  ✓ Loaded model: {encoder.model_name}")
    print(f"  ✓ Hidden size: {encoder.hidden_size}")
    print(f"  ✓ Device: {encoder.device}")
    
    # Test with dummy audio
    print("\n[2] Testing with dummy audio (1 second @ 16kHz)...")
    dummy_audio = torch.randn(16000)  # 1 second at 16kHz
    
    with torch.no_grad():
        outputs = encoder(dummy_audio, sampling_rate=16000, return_dict=True)
    
    print(f"  ✓ Input shape: {dummy_audio.shape}")
    print(f"  ✓ Output hidden_states shape: {outputs['hidden_states'].shape}")
    print(f"  ✓ Output attention_mask shape: {outputs['attention_mask'].shape}")
    print(f"  ✓ Number of audio tokens: {outputs['num_tokens']}")
    
    # Test with batch
    print("\n[3] Testing with batch of 3 audio samples...")
    batch_audio = [torch.randn(16000) for _ in range(3)]
    
    with torch.no_grad():
        batch_outputs = encoder(batch_audio, sampling_rate=16000, return_dict=True)
    
    print(f"  ✓ Batch output shape: {batch_outputs['hidden_states'].shape}")
    print(f"  ✓ Expected: [3, num_tokens, {encoder.hidden_size}]")
    
    # Test with longer audio
    print("\n[4] Testing with longer audio (5 seconds @ 16kHz)...")
    long_audio = torch.randn(80000)  # 5 seconds
    
    with torch.no_grad():
        long_outputs = encoder(long_audio, sampling_rate=16000, return_dict=True)
    
    print(f"  ✓ Long audio output shape: {long_outputs['hidden_states'].shape}")
    print(f"  ✓ More tokens for longer audio: {long_outputs['num_tokens']} tokens")
    
    # Verify output dimension
    print("\n[5] Verifying output properties...")
    assert outputs['hidden_states'].shape[-1] == encoder.output_dim, "Output dimension mismatch"
    assert outputs['hidden_states'].dtype == torch.float32, "Expected float32 output"
    print(f"  ✓ Output dimension correct: {encoder.output_dim}")
    print(f"  ✓ Output dtype correct: {outputs['hidden_states'].dtype}")
    
    print("\n" + "=" * 60)
    print("✓ All AudioEncoder tests passed!")
    print("=" * 60)
    
    return encoder


if __name__ == "__main__":
    test_audio_encoder()
