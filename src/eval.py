#!/usr/bin/env /usr/bin/python3
"""
Evaluation Harness for MiniCPM-AV on MUSIC-AVQA-v2.0
Implements real model inference using forward() pass and logit decoding.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig


def setup_lora_for_eval(model: MiniCPMAV, lora_rank: int = 16, lora_alpha: int = 32) -> MiniCPMAV:
    """
    Apply LoRA to the LLM component for evaluation.
    Must match the training configuration exactly.
    """
    print(f"[Eval] Setting up LoRA (r={lora_rank}, alpha={lora_alpha})...")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # LoRA configuration - MUST match training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    
    # Apply LoRA to model.minicpm.llm
    model.minicpm.llm = get_peft_model(model.minicpm.llm, lora_config)
    print("[Eval] Applied LoRA to LLM backbone")
    
    # Unfreeze audio components (they have actual trained weights)
    for param in model.audio_projector.parameters():
        param.requires_grad = True
    if model.audio_compressor is not None:
        for param in model.audio_compressor.parameters():
            param.requires_grad = True
    for param in model.modality_type_embeddings.parameters():
        param.requires_grad = True
    
    return model
from data_loader import MUSICAVQADataset, AVQACollator, create_dataloaders


class AVQAEvaluator:
    """
    Evaluator for Audio-Visual Question Answering with real model inference.
    Uses forward() pass and logit decoding instead of generate() to avoid
    transformers compatibility issues.
    """
    
    def __init__(
        self,
        model: MiniCPMAV,
        tokenizer: AutoTokenizer,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Answer vocabulary from MUSIC-AVQA dataset
        self.answer_vocab = {
            'indoor': 0, 'no': 1, 'piano': 2, 'saxophone': 3, 'three': 4,
            'outdoor': 5, 'yes': 6, 'violin': 7, 'two': 8, 'one': 9,
            'guitar': 10, 'clapping': 11, 'four': 12, 'five': 13, 'zero': 14
        }
        self.id2answer = {v: k for k, v in self.answer_vocab.items()}
        
        # Token IDs for each answer
        self.answer_token_ids = {}
        for answer in self.answer_vocab.keys():
            tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            if tokens:
                self.answer_token_ids[answer] = tokens[0]  # Use first token
    
    def predict_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        modality: str = "audio_vision"
    ) -> str:
        """
        Predict answer using token generation.
        
        Args:
            input_ids: Tokenized question [seq_len]
            attention_mask: Attention mask [seq_len]
            audio: Audio waveform (optional)
            images: Image tensor (optional)
            modality: Modality mode
            
        Returns:
            Predicted answer string
        """
        with torch.no_grad():
            # Prepare inputs based on modality
            use_audio = audio is not None and modality in ['audio_only', 'audio_vision']
            use_vision = images is not None and modality in ['vision_only', 'audio_vision']
            
            # Encode audio if needed
            audio_embeds = None
            audio_mask = None
            if use_audio:
                audio_outputs = self.model.encode_audio(audio)
                audio_embeds = audio_outputs['audio_embeds']
                audio_mask = audio_outputs['audio_mask']
            
            # Encode vision if needed
            vision_embeds = None
            vision_mask = None
            if use_vision:
                pixel_values = self.model._prepare_images(images)
                vision_features = self.model.minicpm.vpm.forward_features(pixel_values)
                vision_embeds = self.model.minicpm.resampler(vision_features)
                
                batch_size_v = vision_embeds.shape[0]
                num_vision_tokens = vision_embeds.shape[1]
                vision_modality_ids = torch.full(
                    (batch_size_v, num_vision_tokens), 1,
                    dtype=torch.long, device=self.device
                )
                vision_embeds = vision_embeds + self.model.modality_type_embeddings(vision_modality_ids)
                vision_mask = torch.ones(
                    (batch_size_v, num_vision_tokens),
                    dtype=torch.long, device=self.device
                )
            
            # Get text embeddings
            text_embeds = self.model.minicpm.llm.get_input_embeddings()(input_ids.unsqueeze(0))
            text_modality_ids = torch.zeros_like(input_ids.unsqueeze(0))
            text_embeds = text_embeds + self.model.modality_type_embeddings(text_modality_ids)
            
            # Concatenate embeddings
            combined_embeds = []
            combined_mask = []
            
            if audio_embeds is not None:
                combined_embeds.append(audio_embeds)
                combined_mask.append(audio_mask)
            
            if vision_embeds is not None:
                combined_embeds.append(vision_embeds)
                combined_mask.append(vision_mask)
            
            combined_embeds.append(text_embeds)
            combined_mask.append(attention_mask.unsqueeze(0))
            
            inputs_embeds = torch.cat(combined_embeds, dim=1)
            combined_attention_mask = torch.cat(combined_mask, dim=1)
            
            # Generate tokens using greedy decoding
            # Start with the input embeddings
            generated_ids = []
            max_new_tokens = 10  # Limit generation length
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model.minicpm.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=combined_attention_mask,
                    return_dict=True,
                    use_cache=False  # Disable cache to avoid compatibility issues
                )
                
                # Get logits for next token
                next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
                
                # Get most likely token
                next_token_id = torch.argmax(next_token_logits).item()
                
                # Check for EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token_id)
                
                # Prepare next input embedding
                next_token_embed = self.model.minicpm.llm.get_input_embeddings()(
                    torch.tensor([[next_token_id]], device=self.device)
                )
                # Add modality embedding (text = 0)
                next_modality_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                next_token_embed = next_token_embed + self.model.modality_type_embeddings(next_modality_ids)
                
                # Append to inputs
                inputs_embeds = torch.cat([inputs_embeds, next_token_embed], dim=1)
                combined_attention_mask = torch.cat([
                    combined_attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=self.device)
                ], dim=1)
            
            # Decode generated tokens
            if generated_ids:
                predicted_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                predicted_text = predicted_text.strip().lower()
                
                # Extract answer from generated text
                # Look for known answer tokens in the output
                for answer in self.answer_vocab.keys():
                    if answer in predicted_text:
                        return answer
                
                # If no known answer found, return the generated text (first word)
                first_word = predicted_text.split()[0] if predicted_text else "unknown"
                return first_word
            
            return "unknown"
    
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        modality: str = "audio_vision"
    ) -> Dict[str, List]:
        """
        Evaluate a single batch using real model inference.
        
        Args:
            batch: Batched data
            modality: One of 'audio_only', 'vision_only', 'audio_vision'
            
        Returns:
            Dictionary with predictions and references
        """
        questions = batch.get('questions', [])
        references = batch.get('answers', [])
        batch_size = len(questions)
        
        audio = batch.get('audio') if modality in ['audio_only', 'audio_vision'] else None
        images = batch.get('images') if modality in ['vision_only', 'audio_vision'] else None
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        
        predictions = []
        
        for i in range(batch_size):
            try:
                sample_input_ids = input_ids[i] if input_ids is not None else None
                sample_attention_mask = attention_mask[i] if attention_mask is not None else None
                
                sample_audio = None
                if audio is not None:
                    if isinstance(audio, list):
                        sample_audio = audio[i] if i < len(audio) else None
                    elif isinstance(audio, torch.Tensor):
                        sample_audio = audio[i:i+1] if audio.dim() > 1 else audio
                
                sample_images = None
                if images is not None:
                    if isinstance(images, list):
                        sample_images = images[i] if i < len(images) else None
                    elif isinstance(images, torch.Tensor):
                        sample_images = images[i:i+1] if images.dim() > 3 else images
                
                if sample_input_ids is not None:
                    pred = self.predict_answer(
                        input_ids=sample_input_ids.to(self.device),
                        attention_mask=sample_attention_mask.to(self.device) if sample_attention_mask is not None else torch.ones_like(sample_input_ids).to(self.device),
                        audio=sample_audio.to(self.device) if sample_audio is not None and isinstance(sample_audio, torch.Tensor) else sample_audio,
                        images=sample_images.to(self.device) if sample_images is not None and isinstance(sample_images, torch.Tensor) else sample_images,
                        modality=modality
                    )
                else:
                    pred = "unknown"
                
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error generating prediction for sample {i}: {e}")
                predictions.append("error")
        
        return {
            'predictions': predictions,
            'references': references,
            'questions': questions
        }
    
    def compute_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute exact match accuracy."""
        assert len(predictions) == len(references)
        
        correct = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.lower().strip() == ref.lower().strip()
        )
        
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def compute_per_type_accuracy(
        self,
        predictions: List[str],
        references: List[str],
        questions: List[str]
    ) -> Dict[str, Dict]:
        """Compute accuracy broken down by question type."""
        type_predictions = defaultdict(list)
        type_references = defaultdict(list)
        
        for pred, ref, q in zip(predictions, references, questions):
            q_type = self._classify_question_type(q)
            type_predictions[q_type].append(pred)
            type_references[q_type].append(ref)
        
        results = {}
        for q_type in type_predictions:
            results[q_type] = self.compute_accuracy(
                type_predictions[q_type],
                type_references[q_type]
            )
        
        return results
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question into type."""
        q_lower = question.lower()
        
        if 'how many' in q_lower:
            return 'counting'
        elif 'where' in q_lower:
            return 'location'
        elif any(w in q_lower for w in ['is', 'are', 'does', 'do']):
            return 'yes_no'
        elif any(inst in q_lower for inst in ['piano', 'violin', 'guitar', 'saxophone']):
            return 'instrument'
        else:
            return 'other'
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        modality: str = "audio_vision",
        max_samples: Optional[int] = None
    ) -> Dict:
        """Evaluate entire dataset."""
        all_predictions = []
        all_references = []
        all_questions = []
        
        num_batches = len(dataloader)
        if max_samples:
            num_batches = min(num_batches, max_samples // dataloader.batch_size + 1)
        
        print(f"\nEvaluating with modality: {modality}")
        print(f"Batches to process: {num_batches}")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Eval {modality}")):
            if max_samples and len(all_predictions) >= max_samples:
                break
            
            results = self.evaluate_batch(batch, modality)
            
            all_predictions.extend(results['predictions'])
            all_references.extend(results['references'])
            all_questions.extend(results['questions'])
        
        if max_samples:
            all_predictions = all_predictions[:max_samples]
            all_references = all_references[:max_samples]
            all_questions = all_questions[:max_samples]
        
        overall_metrics = self.compute_accuracy(all_predictions, all_references)
        per_type_metrics = self.compute_per_type_accuracy(
            all_predictions, all_references, all_questions
        )
        
        confusion = self._compute_confusion_matrix(all_predictions, all_references)
        
        return {
            'modality': modality,
            'overall': overall_metrics,
            'per_type': dict(per_type_metrics),
            'confusion_matrix': confusion,
            'num_samples': len(all_predictions),
            'predictions': all_predictions[:20],
            'references': all_references[:20]
        }
    
    def _compute_confusion_matrix(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """Compute confusion matrix data."""
        confusion = defaultdict(lambda: defaultdict(int))
        
        for pred, ref in zip(predictions, references):
            confusion[ref][pred] += 1
        
        result = {}
        for ref in confusion:
            result[ref] = dict(confusion[ref])
        
        return result


def run_modality_ablation(
    model: MiniCPMAV,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    max_samples: Optional[int] = None
) -> Dict:
    """Run modality ablation study."""
    evaluator = AVQAEvaluator(model, tokenizer)
    
    results = {}
    
    print("\n" + "="*60)
    print("Modality: Audio Only (Blind)")
    print("="*60)
    results['audio_only'] = evaluator.evaluate_dataset(
        dataloader, modality='audio_only', max_samples=max_samples
    )
    
    print("\n" + "="*60)
    print("Modality: Vision Only (Deaf)")
    print("="*60)
    results['vision_only'] = evaluator.evaluate_dataset(
        dataloader, modality='vision_only', max_samples=max_samples
    )
    
    print("\n" + "="*60)
    print("Modality: Audio + Vision (Full)")
    print("="*60)
    results['audio_vision'] = evaluator.evaluate_dataset(
        dataloader, modality='audio_vision', max_samples=max_samples
    )
    
    return results


def print_results(results: Dict):
    """Print evaluation results in formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for modality, result in results.items():
        print(f"\n{modality.upper().replace('_', ' ')}:")
        print(f"  Accuracy: {result['overall']['accuracy']:.4f}")
        print(f"  Correct: {result['overall']['correct']}/{result['overall']['total']}")
        print(f"  Samples: {result['num_samples']}")
        
        if 'per_type' in result:
            print("  Per-type accuracy:")
            for q_type, metrics in result['per_type'].items():
                print(f"    {q_type}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable[key] = {
                k: v for k, v in value.items()
                if k not in ['predictions', 'references']
            }
            if 'predictions' in value:
                serializable[key]['sample_predictions'] = list(zip(
                    value['predictions'][:10],
                    value['references'][:10]
                ))
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main(args):
    """Main evaluation function."""
    print("="*60)
    print("MiniCPM-AV Evaluation - Real Model Inference")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[2] Initializing MiniCPM-AV model...")
    config = MiniCPMAVConfig(
        use_audio_compression=args.use_audio_compression,
        num_audio_tokens=args.num_audio_tokens
    )
    
    model = MiniCPMAV(config=config)
    
    # Setup LoRA before loading checkpoint (must match training config)
    print("\n[2a] Setting up LoRA configuration...")
    model = setup_lora_for_eval(model, lora_rank=16, lora_alpha=32)
    
    # Load checkpoint
    lora_weights_found = False
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        model.load_pretrained(args.checkpoint_path)
        # Check if LoRA weights exist separately
        lora_path = os.path.join(args.checkpoint_path, 'lora_adapter')
        if os.path.exists(lora_path):
            from peft import PeftModel
            model.minicpm.llm = PeftModel.from_pretrained(model.minicpm.llm, lora_path)
            lora_weights_found = True
    elif args.use_hf_hub:
        print(f"Loading checkpoint from HuggingFace Hub: {args.hf_model_id}")
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            # List files in repo to check for LoRA weights
            repo_files = list_repo_files(args.hf_model_id)
            
            checkpoint_path = hf_hub_download(
                repo_id=args.hf_model_id,
                filename=args.hf_checkpoint_file,
                cache_dir=args.hf_cache_dir
            )
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.audio_projector.load_state_dict(checkpoint['audio_projector'])
            if checkpoint.get('audio_compressor') and model.audio_compressor:
                model.audio_compressor.load_state_dict(checkpoint['audio_compressor'])
            model.modality_type_embeddings.load_state_dict(checkpoint['modality_embeddings'])
            print(f"  ✓ Loaded audio components from HuggingFace Hub")
            
            # Check for LoRA weights
            if any('lora' in f.lower() or 'adapter' in f.lower() for f in repo_files):
                print(f"  ⚠ LoRA weights found in repo but not loaded (manual integration needed)")
                lora_weights_found = True
            else:
                print(f"  ⚠ WARNING: No LoRA weights found in checkpoint!")
                print(f"     The model was trained with LoRA but weights were not saved.")
                print(f"     Evaluation will use untrained base model.")
        except Exception as e:
            print(f"  ⚠ Could not load from HF Hub: {e}")
            print("  Using base MiniCPM-V (no trained checkpoint)")
    else:
        print("Using base MiniCPM-V (no trained checkpoint)")
    
    if not lora_weights_found:
        print("\n" + "="*60)
        print("WARNING: LoRA weights not found!")
        print("="*60)
        print("The checkpoint only contains audio projector weights.")
        print("The LLM LoRA weights (trained for QA) are missing.")
        print("Predictions will be based on untrained model behavior.")
        print("="*60 + "\n")
    
    model.to(device)
    model.eval()
    
    print("\n[3] Creating dataloader...")
    _, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_encoder=model.audio_encoder,
        tokenizer=tokenizer,
        max_train_samples=0
    )
    print(f"  Val batches: {len(val_loader)}")
    
    print("\n[4] Running evaluation...")
    
    if args.ablation:
        results = run_modality_ablation(
            model, tokenizer, val_loader,
            max_samples=args.max_samples
        )
    else:
        evaluator = AVQAEvaluator(model, tokenizer, device)
        results = {
            args.modality: evaluator.evaluate_dataset(
                val_loader,
                modality=args.modality,
                max_samples=args.max_samples
            )
        }
    
    print_results(results)
    
    if args.output_file:
        save_results(results, args.output_file)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MiniCPM-AV with real inference')
    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to local model checkpoint')
    parser.add_argument('--use_hf_hub', action='store_true', default=True,
                        help='Load checkpoint from HuggingFace Hub')
    parser.add_argument('--hf_model_id', type=str, default='gvij/minicpm-av-music-avqa',
                        help='HuggingFace model ID')
    parser.add_argument('--hf_checkpoint_file', type=str, default='best_model/audio_components.pt',
                        help='Checkpoint file in HF repo')
    parser.add_argument('--hf_cache_dir', type=str, default='./checkpoints/hf_cache',
                        help='Cache directory for HF downloads')
    parser.add_argument('--use_audio_compression', action='store_true', default=True,
                        help='Use audio token compression')
    parser.add_argument('--num_audio_tokens', type=int, default=50,
                        help='Number of audio tokens')
    
    parser.add_argument('--modality', type=str, default='audio_vision',
                        choices=['audio_only', 'vision_only', 'audio_vision'],
                        help='Evaluation modality')
    parser.add_argument('--ablation', action='store_true', default=True,
                        help='Run modality ablation study')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate (None for all)')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    parser.add_argument('--output_file', type=str, default='./results/eval_results_real.json',
                        help='Output file for results')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    main(args)
