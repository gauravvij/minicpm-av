#!/usr/bin/env /usr/bin/python3
"""
Audio Projection Module for MiniCPM-AV
Projects Moonshine audio features (288-dim) to MiniCPM-V vision input dimension.
"""

import torch
import torch.nn as nn
from typing import Optional


class AudioProjector(nn.Module):
    """
    Projection layer to align audio features with vision model input space.
    
    Moonshine outputs 288-dim features, but MiniCPM-V's vision encoder
    produces features that get projected to the LLM's input space.
    Based on MiniCPM-V architecture, the projection target is 2304-dim
    (the LLM's hidden size).
    
    Args:
        audio_dim: Input dimension from audio encoder (Moonshine: 288)
        output_dim: Output dimension for LLM input (MiniCPM-V: 2304)
        use_layernorm: Whether to apply LayerNorm for stability
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        audio_dim: int = 288,
        output_dim: int = 2304,
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Linear projection
        self.projection = nn.Linear(audio_dim, output_dim, bias=True)
        
        # LayerNorm for stability (optional but recommended)
        self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights with small values for stability."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to LLM input space.
        
        Args:
            audio_features: Tensor of shape [batch, seq_len, audio_dim]
            
        Returns:
            Projected features of shape [batch, seq_len, output_dim]
        """
        # Linear projection
        projected = self.projection(audio_features)
        
        # Apply LayerNorm if enabled
        if self.layernorm is not None:
            projected = self.layernorm(projected)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            projected = self.dropout(projected)
        
        return projected
    
    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.output_dim


class AudioTokenCompressor(nn.Module):
    """
    Optional token compressor to reduce audio token count.
    Moonshine produces ~200 tokens for typical audio, which can be
    compressed to ~50 tokens for efficiency.
    
    Args:
        num_tokens: Number of output tokens after compression
        input_dim: Dimension of input features
    """
    
    def __init__(self, num_tokens: int = 50, input_dim: int = 2304):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        
        # Learnable query tokens for cross-attention compression
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, input_dim))
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Compress audio tokens using cross-attention.
        
        Args:
            audio_features: [batch, seq_len, input_dim]
            
        Returns:
            Compressed features: [batch, num_tokens, input_dim]
        """
        batch_size = audio_features.shape[0]
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(queries, audio_features, audio_features)
        attn_out = self.norm1(attn_out + queries)
        
        # Feed-forward
        ffn_out = self.ffn(attn_out)
        output = self.norm2(ffn_out + attn_out)
        
        return output


def test_audio_projector():
    """Test script for AudioProjector."""
    print("=" * 60)
    print("Testing AudioProjector Module")
    print("=" * 60)
    
    # Test 1: Basic projection
    print("\n[1] Testing basic projection (288 -> 2304)...")
    projector = AudioProjector(audio_dim=288, output_dim=2304, use_layernorm=True)
    
    # Create dummy audio features
    batch_size, seq_len = 2, 10
    audio_features = torch.randn(batch_size, seq_len, 288)
    
    # Forward pass
    projected = projector(audio_features)
    
    print(f"  ✓ Input shape: {audio_features.shape}")
    print(f"  ✓ Output shape: {projected.shape}")
    print(f"  ✓ Expected: [{batch_size}, {seq_len}, 2304]")
    assert projected.shape == (batch_size, seq_len, 2304), "Shape mismatch"
    
    # Test 2: Verify output properties
    print("\n[2] Verifying output properties...")
    print(f"  ✓ Output mean: {projected.mean().item():.4f}")
    print(f"  ✓ Output std: {projected.std().item():.4f}")
    print(f"  ✓ Output dtype: {projected.dtype}")
    
    # Test 3: Without LayerNorm
    print("\n[3] Testing without LayerNorm...")
    projector_no_ln = AudioProjector(audio_dim=288, output_dim=2304, use_layernorm=False)
    projected_no_ln = projector_no_ln(audio_features)
    print(f"  ✓ Output shape: {projected_no_ln.shape}")
    assert projected_no_ln.shape == (batch_size, seq_len, 2304)
    
    # Test 4: With dropout
    print("\n[4] Testing with dropout (0.1)...")
    projector_dropout = AudioProjector(audio_dim=288, output_dim=2304, dropout=0.1)
    projector_dropout.train()  # Enable dropout
    projected_dropout = projector_dropout(audio_features)
    print(f"  ✓ Output shape: {projected_dropout.shape}")
    assert projected_dropout.shape == (batch_size, seq_len, 2304)
    
    # Test 5: Token compressor (optional)
    print("\n[5] Testing AudioTokenCompressor (10 -> 5 tokens)...")
    compressor = AudioTokenCompressor(num_tokens=5, input_dim=2304)
    compressed = compressor(projected)
    print(f"  ✓ Input tokens: {projected.shape[1]}")
    print(f"  ✓ Output tokens: {compressed.shape[1]}")
    print(f"  ✓ Output shape: {compressed.shape}")
    assert compressed.shape == (batch_size, 5, 2304)
    
    # Test 6: Integration test
    print("\n[6] Integration: AudioEncoder -> Projector -> Compressor")
    from audio_encoder import AudioEncoder
    
    encoder = AudioEncoder(freeze_encoder=True)
    dummy_audio = torch.randn(16000)
    
    with torch.no_grad():
        audio_out = encoder(dummy_audio, return_dict=True)
        audio_features = audio_out['hidden_states']
        
        # Project
        projected_features = projector(audio_features)
        
        # Compress
        compressed_features = compressor(projected_features)
    
    print(f"  ✓ Audio encoder output: {audio_features.shape}")
    print(f"  ✓ After projection: {projected_features.shape}")
    print(f"  ✓ After compression: {compressed_features.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All AudioProjector tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_audio_projector()
