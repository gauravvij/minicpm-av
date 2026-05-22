#!/usr/bin/env /usr/bin/python3
"""
MiniCPM-AV: Audio-Enhanced MiniCPM-V Model
Extends MiniCPM-V with audio understanding capability via Moonshine ASR integration.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass

from audio_encoder import AudioEncoder
from audio_projector import AudioProjector, AudioTokenCompressor


@dataclass
class MiniCPMAVConfig:
    """Configuration for MiniCPM-AV model."""
    # Model paths
    minicpm_model_name: str = "openbmb/MiniCPM-V"
    moonshine_model_name: str = "UsefulSensors/moonshine-tiny"
    
    # Audio settings
    audio_dim: int = 288  # Moonshine output
    projection_dim: int = 2304  # MiniCPM-V hidden size
    use_audio_compression: bool = True
    num_audio_tokens: int = 50  # After compression
    
    # Training settings
    freeze_vision_encoder: bool = True
    freeze_audio_encoder: bool = True
    freeze_llm: bool = False  # Will use LoRA instead
    
    # Modality tokens
    audio_token_id: int = 151646  # Special token for audio (example)
    vision_token_id: int = 151647  # Special token for vision


class MiniCPMAV(nn.Module):
    """
    Audio-Enhanced MiniCPM-V for Audio-Visual Question Answering.
    
    Architecture:
        Audio -> Moonshine Encoder -> Projection -> [Audio Tokens]
        Image -> SigLIP Vision Encoder -> [Vision Tokens]
        [Audio Tokens] + [Vision Tokens] + Text -> MiniCPM-V LLM -> Answer
    """
    
    def __init__(self, config: Optional[MiniCPMAVConfig] = None):
        super().__init__()
        
        self.config = config or MiniCPMAVConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load MiniCPM-V (vision-language backbone)
        print("[MiniCPMAV] Loading MiniCPM-V...")
        self.minicpm = AutoModel.from_pretrained(
            self.config.minicpm_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.minicpm_model_name,
            trust_remote_code=True
        )
        
        # Move to device
        self.minicpm.to(self.device)
        
        # Freeze vision encoder if requested
        if self.config.freeze_vision_encoder:
            self._freeze_vision_encoder()
        
        # Initialize audio encoder
        print("[MiniCPMAV] Loading audio encoder...")
        self.audio_encoder = AudioEncoder(
            model_name=self.config.moonshine_model_name,
            freeze_encoder=self.config.freeze_audio_encoder,
            device=self.device
        )
        
        # Initialize audio projector (288 -> 2304)
        print("[MiniCPMAV] Initializing audio projector...")
        self.audio_projector = AudioProjector(
            audio_dim=self.config.audio_dim,
            output_dim=self.config.projection_dim,
            use_layernorm=True
        )
        
        # Optional: Token compressor
        if self.config.use_audio_compression:
            print("[MiniCPMAV] Initializing audio token compressor...")
            self.audio_compressor = AudioTokenCompressor(
                num_tokens=self.config.num_audio_tokens,
                input_dim=self.config.projection_dim
            )
        else:
            self.audio_compressor = None
        
        # Modality type embeddings (to help model distinguish audio vs vision tokens)
        self.modality_type_embeddings = nn.Embedding(3, self.config.projection_dim)  # 0=text, 1=vision, 2=audio
        
        print(f"[MiniCPMAV] Model initialized on {self.device}")
        print(f"  - Audio tokens: {self.config.num_audio_tokens if self.config.use_audio_compression else 'dynamic'}")
        print(f"  - Projection: {self.config.audio_dim} -> {self.config.projection_dim}")
    
    def _freeze_vision_encoder(self):
        """Freeze the vision encoder in MiniCPM-V."""
        if hasattr(self.minicpm, 'vision_model'):
            for param in self.minicpm.vision_model.parameters():
                param.requires_grad = False
            print("[MiniCPMAV] Frozen vision encoder")
        elif hasattr(self.minicpm, 'vpm'):  # Alternative attribute name
            for param in self.minicpm.vpm.parameters():
                param.requires_grad = False
            print("[MiniCPMAV] Frozen vision encoder (vpm)")
    
    def encode_audio(self, audio: Union[torch.Tensor, List[torch.Tensor]], 
                     sampling_rate: int = 16000) -> Dict[str, torch.Tensor]:
        """
        Encode audio to tokens compatible with MiniCPM-V.
        
        Args:
            audio: Raw audio waveform(s) [batch, samples] or list of [samples]
            sampling_rate: Audio sampling rate
            
        Returns:
            Dictionary with:
                - audio_embeds: [batch, num_tokens, projection_dim]
                - audio_mask: [batch, num_tokens]
                - modality_ids: [batch, num_tokens] (all 2 for audio)
        """
        # Get audio features from Moonshine
        audio_outputs = self.audio_encoder(
            audio, 
            sampling_rate=sampling_rate, 
            return_dict=True
        )
        
        audio_features = audio_outputs['hidden_states']  # [batch, seq_len, 288]
        audio_mask = audio_outputs['attention_mask']
        
        # Project to LLM dimension
        audio_embeds = self.audio_projector(audio_features)  # [batch, seq_len, 2304]
        
        # Compress tokens if enabled
        if self.audio_compressor is not None:
            audio_embeds = self.audio_compressor(audio_embeds)  # [batch, num_audio_tokens, 2304]
            audio_mask = torch.ones(
                audio_embeds.shape[:2], 
                dtype=torch.long, 
                device=self.device
            )
        
        # Add modality type embedding (2 = audio)
        batch_size, num_tokens = audio_embeds.shape[:2]
        modality_ids = torch.full(
            (batch_size, num_tokens), 
            2, 
            dtype=torch.long, 
            device=self.device
        )
        audio_embeds = audio_embeds + self.modality_type_embeddings(modality_ids)
        
        return {
            'audio_embeds': audio_embeds,
            'audio_mask': audio_mask,
            'modality_ids': modality_ids
        }
    
    def encode_image(self, images: Union[torch.Tensor, List]) -> Dict[str, torch.Tensor]:
        """
        Encode images using MiniCPM-V's vision encoder.
        
        Args:
            images: PIL Images or tensors
            
        Returns:
            Dictionary with vision_embeds and vision_mask
        """
        # Use MiniCPM-V's built-in image encoding
        # This is a simplified version - actual implementation would use
        # the model's vision encoder directly
        
        # For now, return placeholder that will be replaced with actual vision encoding
        # In practice, this would call self.minicpm.get_vision_embedding() or similar
        
        # Placeholder: assume vision_embeds are already computed
        # In real implementation, integrate with MiniCPM-V's vision processing
        
        # Return structure for compatibility
        return {
            'vision_embeds': None,  # Will be filled by MiniCPM-V
            'vision_mask': None,
            'modality_ids': None
        }
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[Union[torch.Tensor, List]] = None,
        audio: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting audio, vision, and text inputs.
        
        Architecture:
            1. Encode audio -> audio_embeds [batch, num_audio_tokens, 2304]
            2. Encode images -> vision_embeds [batch, num_vision_tokens, 2304]
            3. Get text embeddings -> text_embeds [batch, seq_len, 2304]
            4. Concatenate: [audio] + [vision] + [text]
            5. Pass through LLM
            6. Compute loss if labels provided
        
        Args:
            input_ids: Text token IDs [batch, seq_len]
            images: Images (optional) - PIL Images or tensors
            audio: Audio waveforms (optional) [batch, samples] or list of [samples]
            attention_mask: Attention mask for text [batch, seq_len]
            labels: Labels for language modeling loss [batch, seq_len]
            
        Returns:
            Dictionary with logits, loss (if labels provided), and hidden states
        """
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        
        # Step 1: Encode audio if provided
        audio_embeds = None
        audio_mask = None
        if audio is not None:
            audio_outputs = self.encode_audio(audio)
            audio_embeds = audio_outputs['audio_embeds']  # [batch, num_audio_tokens, 2304]
            audio_mask = audio_outputs['audio_mask']      # [batch, num_audio_tokens]
        
        # Step 2: Encode images if provided
        vision_embeds = None
        vision_mask = None
        if images is not None:
            # Use MiniCPM-V's vision encoding pipeline
            # First get vision embeddings from vpm
            pixel_values = self._prepare_images(images)  # Convert to tensors if needed
            
            with torch.no_grad() if self.config.freeze_vision_encoder else torch.enable_grad():
                # Get vision features from SigLIP
                vision_features = self.minicpm.vpm.forward_features(pixel_values)  # [batch, num_patches, vision_dim]
                
                # Resample to fixed number of tokens
                vision_embeds = self.minicpm.resampler(vision_features)  # [batch, num_vision_tokens, 2304]
                
                # Add modality type embedding (1 = vision)
                batch_size_v = vision_embeds.shape[0]
                num_vision_tokens = vision_embeds.shape[1]
                vision_modality_ids = torch.full(
                    (batch_size_v, num_vision_tokens),
                    1,  # 1 = vision
                    dtype=torch.long,
                    device=self.device
                )
                vision_embeds = vision_embeds + self.modality_type_embeddings(vision_modality_ids)
                
                vision_mask = torch.ones(
                    (batch_size_v, num_vision_tokens),
                    dtype=torch.long,
                    device=self.device
                )
        
        # Step 3: Get text embeddings
        text_embeds = self.minicpm.llm.get_input_embeddings()(input_ids)  # [batch, seq_len, 2304]
        
        # Add modality type embedding (0 = text)
        text_modality_ids = torch.zeros_like(input_ids)  # 0 = text
        text_embeds = text_embeds + self.modality_type_embeddings(text_modality_ids)
        
        # Step 4: Concatenate embeddings: [audio] + [vision] + [text]
        combined_embeds = []
        combined_mask = []
        
        if audio_embeds is not None:
            combined_embeds.append(audio_embeds)
            combined_mask.append(audio_mask)
        
        if vision_embeds is not None:
            combined_embeds.append(vision_embeds)
            combined_mask.append(vision_mask)
        
        combined_embeds.append(text_embeds)
        combined_mask.append(attention_mask if attention_mask is not None else torch.ones_like(input_ids))
        
        # Concatenate along sequence dimension
        inputs_embeds = torch.cat(combined_embeds, dim=1)  # [batch, total_seq_len, 2304]
        combined_attention_mask = torch.cat(combined_mask, dim=1)  # [batch, total_seq_len]
        
        # Adjust labels to match combined sequence length
        if labels is not None:
            # Shift labels to account for audio and vision tokens
            prefix_len = inputs_embeds.shape[1] - input_ids.shape[1]
            prefix_labels = torch.full(
                (batch_size, prefix_len),
                -100,  # Ignore index for loss computation
                dtype=torch.long,
                device=self.device
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)  # [batch, total_seq_len]
        else:
            combined_labels = None
        
        # Step 5: Pass through LLM
        outputs = self.minicpm.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True,
            **kwargs
        )
        
        # outputs contains: loss, logits, past_key_values, hidden_states, attentions
        if return_dict:
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'audio_embeds': audio_embeds,
                'vision_embeds': vision_embeds,
                'text_embeds': text_embeds,
                'combined_embeds': inputs_embeds,
                'past_key_values': outputs.past_key_values,
                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions,
            }
        else:
            return (outputs.loss, outputs.logits, audio_embeds, vision_embeds, text_embeds)
    
    def _prepare_images(self, images: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        Convert images to tensor format expected by vision encoder.
        
        Args:
            images: PIL Images, tensors, or list of images
            
        Returns:
            Tensor of shape [batch, channels, height, width]
        """
        from torchvision import transforms
        
        if isinstance(images, torch.Tensor):
            # Already a tensor
            if images.dim() == 3:
                images = images.unsqueeze(0)
            return images.to(self.device)
        
        if isinstance(images, list):
            # List of PIL Images or tensors
            processed = []
            transform = transforms.Compose([
                transforms.Resize((448, 448)),  # SigLIP input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            for img in images:
                if isinstance(img, torch.Tensor):
                    processed.append(img)
                else:
                    # PIL Image
                    processed.append(transform(img))
            
            return torch.stack(processed).to(self.device)
        
        # Single PIL Image
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(images).unsqueeze(0).to(self.device)
    
    def generate_with_audio(
        self,
        images: Optional[Union[torch.Tensor, List]] = None,
        audio: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        question: str = "",
        **generation_kwargs
    ) -> str:
        """
        Generate answer given image, audio, and question.
        
        Args:
            images: Image(s) (optional)
            audio: Audio waveform(s) (optional)
            question: Text question
            generation_kwargs: Additional generation parameters
            
        Returns:
            Generated answer string
        """
        # This is a high-level interface for inference
        # Would integrate audio encoding with MiniCPM-V's chat/generate
        
        # Placeholder implementation
        if audio is not None:
            audio_embeds = self.encode_audio(audio)['audio_embeds']
            # Would integrate with MiniCPM-V's generation
            # by prepending audio embeddings to the input
        
        # For now, fall back to standard MiniCPM-V chat if no audio
        if audio is None and images is not None:
            # Use standard MiniCPM-V
            answer, _, _ = self.minicpm.chat(
                image=images,
                msgs=[{"role": "user", "content": question}],
                tokenizer=self.tokenizer,
                **generation_kwargs
            )
            return answer
        
        return "Audio-enhanced generation not yet fully implemented"
    
    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save audio components
        torch.save({
            'audio_projector': self.audio_projector.state_dict(),
            'audio_compressor': self.audio_compressor.state_dict() if self.audio_compressor else None,
            'modality_embeddings': self.modality_type_embeddings.state_dict(),
            'config': self.config
        }, os.path.join(save_path, 'audio_components.pt'))
        
        # Save LoRA adapters if present (PEFT model)
        if hasattr(self.minicpm.llm, 'save_pretrained'):
            # Check if this is a PEFT model with LoRA adapters
            lora_save_path = os.path.join(save_path, 'lora_adapters')
            self.minicpm.llm.save_pretrained(lora_save_path)
            print(f"[MiniCPMAV] Saved LoRA adapters to {lora_save_path}")
        
        # Save MiniCPM-V (optional, can be large)
        # self.minicpm.save_pretrained(os.path.join(save_path, 'minicpm'))
        
        print(f"[MiniCPMAV] Saved checkpoint to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load model checkpoint."""
        import os
        
        checkpoint = torch.load(
            os.path.join(load_path, 'audio_components.pt'),
            map_location=self.device
        )
        
        self.audio_projector.load_state_dict(checkpoint['audio_projector'])
        if checkpoint['audio_compressor'] and self.audio_compressor:
            self.audio_compressor.load_state_dict(checkpoint['audio_compressor'])
        self.modality_type_embeddings.load_state_dict(checkpoint['modality_embeddings'])
        
        print(f"[MiniCPMAV] Loaded checkpoint from {load_path}")


def test_minicpm_av():
    """Test script for MiniCPM-AV model."""
    print("=" * 60)
    print("Testing MiniCPM-AV Model")
    print("=" * 60)
    
    # Test 1: Initialize model
    print("\n[1] Initializing MiniCPM-AV...")
    config = MiniCPMAVConfig(
        use_audio_compression=True,
        num_audio_tokens=50
    )
    model = MiniCPMAV(config=config)
    print(f"  ✓ Model initialized")
    print(f"  ✓ Device: {model.device}")
    
    # Test 2: Audio encoding
    print("\n[2] Testing audio encoding...")
    dummy_audio = torch.randn(16000)  # 1 second
    audio_outputs = model.encode_audio(dummy_audio, sampling_rate=16000)
    
    print(f"  ✓ Audio embeds shape: {audio_outputs['audio_embeds'].shape}")
    print(f"  ✓ Audio mask shape: {audio_outputs['audio_mask'].shape}")
    print(f"  ✓ Modality IDs shape: {audio_outputs['modality_ids'].shape}")
    print(f"  ✓ Expected: [1, 50, 2304] (with compression)")
    
    assert audio_outputs['audio_embeds'].shape == (1, 50, 2304), "Shape mismatch"
    assert audio_outputs['modality_ids'][0, 0].item() == 2, "Audio modality ID should be 2"
    
    # Test 3: Batch audio encoding
    print("\n[3] Testing batch audio encoding...")
    batch_audio = [torch.randn(16000) for _ in range(3)]
    batch_outputs = model.encode_audio(batch_audio, sampling_rate=16000)
    
    print(f"  ✓ Batch audio embeds shape: {batch_outputs['audio_embeds'].shape}")
    print(f"  ✓ Expected: [3, 50, 2304]")
    assert batch_outputs['audio_embeds'].shape == (3, 50, 2304)
    
    # Test 4: Forward pass (simplified)
    print("\n[4] Testing forward pass...")
    outputs = model(audio=dummy_audio)
    print(f"  ✓ Forward pass completed")
    print(f"  ✓ Audio embeds returned: {outputs['audio_embeds'] is not None}")
    
    # Test 5: Model parameters
    print("\n[5] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Total parameters: {total_params / 1e6:.1f}M")
    print(f"  ✓ Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"  ✓ Frozen parameters: {(total_params - trainable_params) / 1e6:.1f}M")
    
    print("\n" + "=" * 60)
    print("✓ All MiniCPM-AV tests passed!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_minicpm_av()
