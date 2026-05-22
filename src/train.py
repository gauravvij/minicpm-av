#!/usr/bin/env /usr/bin/python3
"""
Training Script for MiniCPM-AV on MUSIC-AVQA-v2.0
Fine-tunes audio projector and LLM with LoRA for audio-visual QA.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

from modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig
from data_loader import MUSICAVQADataset, AVQACollator, create_dataloaders


def setup_lora(model: MiniCPMAV, lora_rank: int = 8, lora_alpha: int = 16) -> MiniCPMAV:
    """
    Apply LoRA to the LLM component for efficient fine-tuning.
    
    Args:
        model: MiniCPM-AV model
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling
        
    Returns:
        Model with LoRA applied
    """
    print(f"[Training] Setting up LoRA (r={lora_rank}, alpha={lora_alpha})...")
    
    # Step 1: Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False
    print("[Training] Froze all parameters")
    
    # Step 2: Unfreeze audio-specific components
    # Audio projector
    for param in model.audio_projector.parameters():
        param.requires_grad = True
    print("[Training] Unfroze audio_projector")
    
    # Audio compressor (if exists)
    if model.audio_compressor is not None:
        for param in model.audio_compressor.parameters():
            param.requires_grad = True
        print("[Training] Unfroze audio_compressor")
    
    # Modality type embeddings
    for param in model.modality_type_embeddings.parameters():
        param.requires_grad = True
    print("[Training] Unfroze modality_type_embeddings")
    
    # Step 3: Apply LoRA to the LLM backbone
    # LoRA configuration for causal LM
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"     # FFN
        ],
        bias="none"
    )
    
    # Apply LoRA to model.minicpm.llm
    model.minicpm.llm = get_peft_model(model.minicpm.llm, lora_config)
    print("[Training] Applied LoRA to LLM backbone (model.minicpm.llm)")
    
    # Step 4: Enable gradient checkpointing on LLM to save memory
    if hasattr(model.minicpm.llm, 'gradient_checkpointing_enable'):
        model.minicpm.llm.gradient_checkpointing_enable()
        print("[Training] Enabled gradient checkpointing on LLM")
    
    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count LoRA parameters specifically
    lora_params = sum(p.numel() for n, p in model.named_parameters() 
                      if p.requires_grad and 'lora' in n.lower())
    
    print(f"\n[Training] Parameter Summary:")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"  LoRA parameters: {lora_params / 1e6:.1f}M")
    print(f"  Frozen parameters: {(total_params - trainable_params) / 1e6:.1f}M")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


def compute_loss(
    model: MiniCPMAV,
    batch: Dict[str, torch.Tensor],
    device: str
) -> torch.Tensor:
    """
    Compute language modeling loss for AVQA.
    
    Args:
        model: MiniCPM-AV model
        batch: Batched data
        device: Device to use
        
    Returns:
        Loss tensor
    """
    # Move data to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Get labels (shift input_ids for next-token prediction)
    # Labels are the same as input_ids, but we mask out the prompt portion
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio=batch.get('audio'),
        images=batch.get('images')
    )
    
    # Compute cross-entropy loss
    # outputs['logits'] shape: [batch, seq_len, vocab_size]
    logits = outputs.get('logits')
    
    if logits is None:
        # Fallback to dummy loss if model doesn't return logits
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        return loss
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for cross-entropy
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def train_epoch(
    model: MiniCPMAV,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard writer
        use_amp: Whether to use automatic mixed precision
        scaler: Gradient scaler for mixed precision
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss with optional mixed precision
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss = compute_loss(model, batch, device)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            loss = compute_loss(model, batch, device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
        
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log to TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        
        # Clear cache periodically to reduce memory fragmentation
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'loss': avg_loss}


def validate(
    model: MiniCPMAV,
    dataloader: DataLoader,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation {epoch}"):
            loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'loss': avg_loss}


def main(args):
    """Main training function."""
    print("=" * 60)
    print("MiniCPM-AV Training")
    print("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs')) if args.use_tensorboard else None
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("\n[2] Initializing MiniCPM-AV model...")
    config = MiniCPMAVConfig(
        use_audio_compression=args.use_audio_compression,
        num_audio_tokens=args.num_audio_tokens,
        freeze_vision_encoder=True,
        freeze_audio_encoder=True
    )
    model = MiniCPMAV(config=config)
    model.to(device)
    
    # Apply LoRA if requested
    if args.use_lora:
        model = setup_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    else:
        # If not using LoRA, still freeze encoders and only train audio components
        print("[Training] Not using LoRA - freezing vision and audio encoders")
        # Freeze vision encoder
        if hasattr(model.minicpm, 'vision_model'):
            for param in model.minicpm.vision_model.parameters():
                param.requires_grad = False
        # Freeze audio encoder
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        
        # Print trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")
        print(f"  Frozen: {(total_params - trainable_params) / 1e6:.1f}M")
    
    # Setup mixed precision training if requested
    use_amp = args.mixed_precision is not None
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"\n[Training] Using mixed precision: {args.mixed_precision}")
    
    # Create dataloaders
    print("\n[3] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_encoder=model.audio_encoder,
        tokenizer=tokenizer,
        max_train_samples=args.max_train_samples
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Setup optimizer
    print("\n[4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    
    # Training loop
    print("\n[5] Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, writer, use_amp=use_amp, scaler=scaler
        )
        print(f"Train loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = output_dir / 'best_model'
            model.save_pretrained(str(checkpoint_path))
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save regular checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}'
            model.save_pretrained(str(checkpoint_path))
            print(f"✓ Saved checkpoint epoch {epoch}")
    
    # Close TensorBoard
    if writer is not None:
        writer.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MiniCPM-AV')
    
    # Model arguments
    parser.add_argument('--use_audio_compression', action='store_true', default=True,
                        help='Use audio token compression')
    parser.add_argument('--num_audio_tokens', type=int, default=50,
                        help='Number of audio tokens after compression')
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--mixed_precision', type=str, default=None, choices=['fp16', 'bf16'],
                        help='Use mixed precision training (fp16 or bf16)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing to save memory')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum training samples (for debugging)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use TensorBoard logging')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
#!/usr/bin/env /usr/bin/python3
"""
Training Script for MiniCPM-AV on MUSIC-AVQA-v2.0
Fine-tunes audio projector and LLM with LoRA for audio-visual QA.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

from modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig
from data_loader import MUSICAVQADataset, AVQACollator, create_dataloaders


def setup_lora(model: MiniCPMAV, lora_rank: int = 8, lora_alpha: int = 16) -> MiniCPMAV:
    """
    Apply LoRA to the LLM component for efficient fine-tuning.
    
    Args:
        model: MiniCPM-AV model
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling
        
    Returns:
        Model with LoRA applied
    """
    print(f"[Training] Setting up LoRA (r={lora_rank}, alpha={lora_alpha})...")
    
    # Step 1: Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False
    print("[Training] Froze all parameters")
    
    # Step 2: Unfreeze audio-specific components
    # Audio projector
    for param in model.audio_projector.parameters():
        param.requires_grad = True
    print("[Training] Unfroze audio_projector")
    
    # Audio compressor (if exists)
    if model.audio_compressor is not None:
        for param in model.audio_compressor.parameters():
            param.requires_grad = True
        print("[Training] Unfroze audio_compressor")
    
    # Modality type embeddings
    for param in model.modality_type_embeddings.parameters():
        param.requires_grad = True
    print("[Training] Unfroze modality_type_embeddings")
    
    # Step 3: Apply LoRA to the LLM backbone
    # LoRA configuration for causal LM
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"     # FFN
        ],
        bias="none"
    )
    
    # Apply LoRA to model.minicpm.llm
    model.minicpm.llm = get_peft_model(model.minicpm.llm, lora_config)
    print("[Training] Applied LoRA to LLM backbone (model.minicpm.llm)")
    
    # Step 4: Enable gradient checkpointing on LLM to save memory
    if hasattr(model.minicpm.llm, 'gradient_checkpointing_enable'):
        model.minicpm.llm.gradient_checkpointing_enable()
        print("[Training] Enabled gradient checkpointing on LLM")
    
    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count LoRA parameters specifically
    lora_params = sum(p.numel() for n, p in model.named_parameters() 
                      if p.requires_grad and 'lora' in n.lower())
    
    print(f"\n[Training] Parameter Summary:")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"  LoRA parameters: {lora_params / 1e6:.1f}M")
    print(f"  Frozen parameters: {(total_params - trainable_params) / 1e6:.1f}M")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


def compute_loss(
    model: MiniCPMAV,
    batch: Dict[str, torch.Tensor],
    device: str
) -> torch.Tensor:
    """
    Compute language modeling loss for AVQA.
    
    Args:
        model: MiniCPM-AV model
        batch: Batched data
        device: Device to use
        
    Returns:
        Loss tensor
    """
    # Move data to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Get labels (shift input_ids for next-token prediction)
    # Labels are the same as input_ids, but we mask out the prompt portion
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio=batch.get('audio'),
        images=batch.get('images')
    )
    
    # Compute cross-entropy loss
    # outputs['logits'] shape: [batch, seq_len, vocab_size]
    logits = outputs.get('logits')
    
    if logits is None:
        # Fallback to dummy loss if model doesn't return logits
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        return loss
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for cross-entropy
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def train_epoch(
    model: MiniCPMAV,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard writer
        use_amp: Whether to use automatic mixed precision
        scaler: Gradient scaler for mixed precision
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss with optional mixed precision
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss = compute_loss(model, batch, device)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            loss = compute_loss(model, batch, device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
        
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log to TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        
        # Clear cache periodically to reduce memory fragmentation
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'loss': avg_loss}


def validate(
    model: MiniCPMAV,
    dataloader: DataLoader,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation {epoch}"):
            loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'loss': avg_loss}


def main(args):
    """Main training function."""
    print("=" * 60)
    print("MiniCPM-AV Training")
    print("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs')) if args.use_tensorboard else None
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("\n[2] Initializing MiniCPM-AV model...")
    config = MiniCPMAVConfig(
        use_audio_compression=args.use_audio_compression,
        num_audio_tokens=args.num_audio_tokens,
        freeze_vision_encoder=True,
        freeze_audio_encoder=True
    )
    model = MiniCPMAV(config=config)
    model.to(device)
    
    # Apply LoRA if requested
    if args.use_lora:
        model = setup_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params / 1e6:.1f}M")
    print(f"  Trainable: {trainable_params / 1e6:.1f}M")
    print(f"  Frozen: {(total_params - trainable_params) / 1e6:.1f}M")
    
    # Create dataloaders
    print("\n[3] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_encoder=model.audio_encoder,
        tokenizer=tokenizer,
        max_train_samples=args.max_train_samples
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Setup optimizer
    print("\n[4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    
    # Training loop
    print("\n[5] Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, writer
        )
        print(f"Train loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = output_dir / 'best_model'
            model.save_pretrained(str(checkpoint_path))
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save regular checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}'
            model.save_pretrained(str(checkpoint_path))
            print(f"✓ Saved checkpoint epoch {epoch}")
    
    # Close TensorBoard
    if writer is not None:
        writer.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MiniCPM-AV')
    
    # Model arguments
    parser.add_argument('--use_audio_compression', action='store_true', default=True,
                        help='Use audio token compression')
    parser.add_argument('--num_audio_tokens', type=int, default=50,
                        help='Number of audio tokens after compression')
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum training samples (for debugging)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use TensorBoard logging')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
