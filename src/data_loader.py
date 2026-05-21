#!/usr/bin/env /usr/bin/python3
"""
Data Pipeline for MiniCPM-AV Training on MUSIC-AVQA-v2.0
Loads audio-visual QA data and prepares it for training.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import json


class MUSICAVQADataset(Dataset):
    """
    Dataset for MUSIC-AVQA-v2.0 audio-visual question answering.
    
    Each sample contains:
    - video_id: YouTube video ID
    - question: Text question about audio-visual content
    - answer: Text answer
    - audio_path: Path to audio file (to be downloaded/processed)
    - image_path: Path to video frame/image
    """
    
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.split = split
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load dataset from HuggingFace
        print(f"[MUSICAVQA] Loading {split} split from HuggingFace...")
        try:
            self.dataset = load_dataset(
                "DraculaDragon/MUSIC-AVQA-v2.0",
                split=split,
                cache_dir=self.cache_dir
            )
            print(f"[MUSICAVQA] Loaded {len(self.dataset)} samples")
        except Exception as e:
            print(f"[MUSICAVQA] Error loading dataset: {e}")
            print("[MUSICAVQA] Creating empty dataset for testing...")
            self.dataset = []
        
        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            print(f"[MUSICAVQA] Limited to {max_samples} samples")
        
        # Question type mapping
        self.question_types = {
            "Audio-Visual": "av",
            "Audio": "audio",
            "Visual": "visual"
        }
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        item = self.dataset[idx]
        
        # Extract fields
        sample = {
            'video_id': item.get('video_id', ''),
            'question_id': item.get('question_id', ''),
            'question_type': item.get('type', ''),
            'question': item.get('question_content', ''),
            'answer': item.get('anser', ''),  # Note: dataset has typo 'anser'
            'templ_values': item.get('templ_values', []),
            'audio_instrument': item.get('audio_instrument', ''),
            'video_instrument': item.get('video_instrument', ''),
            'person': item.get('person', '')
        }
        
        return sample
    
    def get_question_type(self, idx: int) -> str:
        """Get the question type (audio, visual, or av)."""
        item = self.dataset[idx]
        q_type = item.get('type', '')
        return self.question_types.get(q_type, 'unknown')
    
    def get_modality_requirements(self, idx: int) -> Dict[str, bool]:
        """
        Determine which modalities are needed for this question.
        
        Returns:
            Dict with 'audio' and 'vision' boolean flags
        """
        q_type = self.get_question_type(idx)
        
        return {
            'audio': q_type in ['av', 'audio'],
            'vision': q_type in ['av', 'visual']
        }


class AVQACollator:
    """
    Collator for batching AVQA samples.
    Handles audio processing, image loading, and tokenization.
    """
    
    def __init__(
        self,
        audio_encoder,
        tokenizer,
        image_processor=None,
        max_length: int = 512,
        audio_sampling_rate: int = 16000
    ):
        self.audio_encoder = audio_encoder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.audio_sampling_rate = audio_sampling_rate
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched tensors ready for model
        """
        # For now, return simplified batch structure
        # Full implementation would:
        # 1. Load and process audio for each sample
        # 2. Load and process images
        # 3. Tokenize questions and answers
        # 4. Create labels for training
        
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        question_types = [item['question_type'] for item in batch]
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize text (questions + answers for training)
        # In practice, we'd create proper input/target sequences
        text_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Placeholder for audio and vision
        # These would be loaded from files in full implementation
        batch_size = len(batch)
        
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'questions': questions,
            'answers': answers,
            'question_types': question_types,
            'batch_size': batch_size,
            # Placeholder for audio/vision - would be actual tensors
            'audio': None,  # Would be [batch, audio_tokens, hidden_dim]
            'images': None  # Would be processed images
        }


def get_answer_mapping(dataset_split) -> Dict[str, int]:
    """
    Create a mapping from answer strings to indices.
    MUSIC-AVQA has a fixed set of possible answers.
    
    Args:
        dataset_split: Dataset split to analyze
        
    Returns:
        Dictionary mapping answer -> index
    """
    answers = set()
    for item in dataset_split:
        answer = item.get('anser', '')
        if answer:
            answers.add(answer)
    
    # Sort for deterministic ordering
    answer_list = sorted(list(answers))
    answer_to_idx = {ans: idx for idx, ans in enumerate(answer_list)}
    
    print(f"[Data] Found {len(answer_list)} unique answers")
    return answer_to_idx


def create_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 4,
    num_workers: int = 2,
    audio_encoder=None,
    tokenizer=None,
    max_train_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory for data storage
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        audio_encoder: Audio encoder model
        tokenizer: Text tokenizer
        max_train_samples: Maximum training samples (for debugging)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MUSICAVQADataset(
        split="train",
        data_dir=data_dir,
        max_samples=max_train_samples
    )
    
    val_dataset = MUSICAVQADataset(
        split="test",  # Using test as validation
        data_dir=data_dir
    )
    
    # Create collator
    collator = AVQACollator(
        audio_encoder=audio_encoder,
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_data_pipeline():
    """Test the data pipeline."""
    print("=" * 60)
    print("Testing Data Pipeline")
    print("=" * 60)
    
    # Test 1: Load dataset
    print("\n[1] Loading MUSIC-AVQA-v2.0 dataset...")
    try:
        dataset = MUSICAVQADataset(split="train", max_samples=10)
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")
        
        # Show first sample
        sample = dataset[0]
        print(f"\n  Sample structure:")
        for key, value in sample.items():
            value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            print(f"    - {key}: {value_str}")
        
        # Show question types
        print(f"\n  Question type: {dataset.get_question_type(0)}")
        print(f"  Modality requirements: {dataset.get_modality_requirements(0)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Check answer distribution
    print("\n[2] Analyzing answer distribution...")
    try:
        if len(dataset) > 0:
            answer_map = get_answer_mapping(dataset.dataset)
            print(f"  ✓ Unique answers: {len(answer_map)}")
            print(f"  ✓ Sample answers: {list(answer_map.keys())[:5]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Test collator
    print("\n[3] Testing collator...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-V",
            trust_remote_code=True
        )
        
        collator = AVQACollator(
            audio_encoder=None,
            tokenizer=tokenizer
        )
        
        # Create mini batch
        batch = [dataset[i] for i in range(min(3, len(dataset)))]
        batched = collator(batch)
        
        print(f"  ✓ Batch input_ids shape: {batched['input_ids'].shape}")
        print(f"  ✓ Batch attention_mask shape: {batched['attention_mask'].shape}")
        print(f"  ✓ Questions: {batched['questions'][:2]}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ Data pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_pipeline()
