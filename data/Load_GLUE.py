#!/usr/bin/env python3
"""
GLUE Benchmark Data Loader for Quantum Models

Loads and preprocesses GLUE benchmark tasks for use with QuantumHydraGated.
Supports all 9 GLUE tasks with automatic tokenization and batching.

Tasks:
- CoLA: Corpus of Linguistic Acceptability (single sentence, binary)
- SST-2: Stanford Sentiment Treebank (single sentence, binary)
- MRPC: Microsoft Research Paraphrase Corpus (sentence pair, binary)
- QQP: Quora Question Pairs (sentence pair, binary)
- STS-B: Semantic Textual Similarity Benchmark (sentence pair, regression)
- MNLI: Multi-Genre Natural Language Inference (sentence pair, 3-class)
- QNLI: Question Natural Language Inference (sentence pair, binary)
- RTE: Recognizing Textual Entailment (sentence pair, binary)
- WNLI: Winograd Natural Language Inference (sentence pair, binary)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

# Try to import datasets library (HuggingFace)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not available. Install with: pip install datasets")

# Try to import tokenizers
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: 'transformers' library not available. Install with: pip install transformers")


# ================================================================================
# GLUE Task Configuration
# ================================================================================

GLUE_TASKS = {
    'cola': {
        'name': 'CoLA',
        'type': 'single',
        'num_labels': 2,
        'metric': 'matthews_correlation',
        'text_columns': ['sentence'],
        'label_column': 'label',
    },
    'sst2': {
        'name': 'SST-2',
        'type': 'single',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['sentence'],
        'label_column': 'label',
    },
    'mrpc': {
        'name': 'MRPC',
        'type': 'pair',
        'num_labels': 2,
        'metric': 'f1',
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
    },
    'qqp': {
        'name': 'QQP',
        'type': 'pair',
        'num_labels': 2,
        'metric': 'f1',
        'text_columns': ['question1', 'question2'],
        'label_column': 'label',
    },
    'stsb': {
        'name': 'STS-B',
        'type': 'pair',
        'num_labels': 1,  # Regression
        'metric': 'pearson',
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
        'is_regression': True,
    },
    'mnli': {
        'name': 'MNLI',
        'type': 'pair',
        'num_labels': 3,
        'metric': 'accuracy',
        'text_columns': ['premise', 'hypothesis'],
        'label_column': 'label',
    },
    'qnli': {
        'name': 'QNLI',
        'type': 'pair',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['question', 'sentence'],
        'label_column': 'label',
    },
    'rte': {
        'name': 'RTE',
        'type': 'pair',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
    },
    'wnli': {
        'name': 'WNLI',
        'type': 'pair',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
    },
}


# ================================================================================
# Simple Tokenizer (fallback when transformers not available)
# ================================================================================

class SimpleTokenizer:
    """
    Simple word-level tokenizer as fallback.
    For production, use a proper tokenizer from transformers.
    """

    def __init__(self, vocab_size: int = 30000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_built = False

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top vocab_size - 4 (for special tokens)
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.vocab_built = True
        print(f"Built vocabulary with {len(self.word2idx)} words")

    def encode(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs."""
        max_len = max_length or self.max_length

        words = text.lower().split()
        token_ids = [self.word2idx.get('[CLS]', 2)]

        for word in words[:max_len - 2]:  # Leave room for [CLS] and [SEP]
            token_ids.append(self.word2idx.get(word, self.word2idx['[UNK]']))

        token_ids.append(self.word2idx.get('[SEP]', 3))

        # Pad to max_length
        attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
        token_ids = token_ids + [0] * (max_len - len(token_ids))

        return {
            'input_ids': torch.tensor(token_ids[:max_len], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:max_len], dtype=torch.long)
        }

    def encode_pair(self, text1: str, text2: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Encode sentence pair to token IDs."""
        max_len = max_length or self.max_length

        words1 = text1.lower().split()
        words2 = text2.lower().split()

        # Calculate space for each sentence (roughly half each, minus special tokens)
        available = max_len - 3  # [CLS], [SEP], [SEP]
        len1 = min(len(words1), available // 2)
        len2 = min(len(words2), available - len1)

        token_ids = [self.word2idx.get('[CLS]', 2)]

        for word in words1[:len1]:
            token_ids.append(self.word2idx.get(word, self.word2idx['[UNK]']))
        token_ids.append(self.word2idx.get('[SEP]', 3))

        for word in words2[:len2]:
            token_ids.append(self.word2idx.get(word, self.word2idx['[UNK]']))
        token_ids.append(self.word2idx.get('[SEP]', 3))

        # Pad
        attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
        token_ids = token_ids + [0] * (max_len - len(token_ids))

        return {
            'input_ids': torch.tensor(token_ids[:max_len], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:max_len], dtype=torch.long)
        }


# ================================================================================
# GLUE Dataset Class
# ================================================================================

class GLUEDataset(Dataset):
    """
    PyTorch Dataset for GLUE benchmark tasks.
    """

    def __init__(
        self,
        task_name: str,
        split: str = 'train',
        tokenizer = None,
        max_length: int = 128,
        cache_dir: Optional[str] = None
    ):
        self.task_name = task_name.lower()
        self.split = split
        self.max_length = max_length

        if self.task_name not in GLUE_TASKS:
            raise ValueError(f"Unknown GLUE task: {task_name}. Available: {list(GLUE_TASKS.keys())}")

        self.task_config = GLUE_TASKS[self.task_name]
        self.is_regression = self.task_config.get('is_regression', False)

        # Load tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            if TOKENIZER_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            else:
                self.tokenizer = SimpleTokenizer(max_length=max_length)

        # Load dataset
        self._load_data(cache_dir)

    def _load_data(self, cache_dir: Optional[str] = None):
        """Load GLUE data from HuggingFace datasets."""
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets library required. Install with: pip install datasets")

        # Handle MNLI special case (matched/mismatched validation)
        if self.task_name == 'mnli':
            if self.split == 'validation':
                self.split = 'validation_matched'
            elif self.split == 'test':
                self.split = 'test_matched'

        # Load dataset
        dataset = load_dataset('glue', self.task_name, cache_dir=cache_dir)

        if self.split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{self.split}' not found. Available: {available_splits}")

        self.data = dataset[self.split]

        # Build vocabulary for simple tokenizer
        if isinstance(self.tokenizer, SimpleTokenizer) and not self.tokenizer.vocab_built:
            all_texts = []
            for item in self.data:
                for col in self.task_config['text_columns']:
                    all_texts.append(item[col])
            self.tokenizer.build_vocab(all_texts)

        print(f"Loaded {self.task_config['name']} {self.split}: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Get text(s)
        text_cols = self.task_config['text_columns']

        if self.task_config['type'] == 'single':
            text = item[text_cols[0]]

            if isinstance(self.tokenizer, SimpleTokenizer):
                encoded = self.tokenizer.encode(text, max_length=self.max_length)
            else:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        else:
            text1 = item[text_cols[0]]
            text2 = item[text_cols[1]]

            if isinstance(self.tokenizer, SimpleTokenizer):
                encoded = self.tokenizer.encode_pair(text1, text2, max_length=self.max_length)
            else:
                encoded = self.tokenizer(
                    text1, text2,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                encoded = {k: v.squeeze(0) for k, v in encoded.items()}

        # Get label
        label_col = self.task_config['label_column']
        label = item[label_col]

        if self.is_regression:
            # Normalize STS-B labels to [0, 1]
            label = torch.tensor(label / 5.0, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        encoded['labels'] = label

        return encoded


# ================================================================================
# Data Loader Factory
# ================================================================================

def load_glue_task(
    task_name: str,
    tokenizer = None,
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load GLUE task data and create DataLoaders.

    Args:
        task_name: Name of GLUE task (cola, sst2, mrpc, etc.)
        tokenizer: Tokenizer instance (default: bert-base-uncased)
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        cache_dir: Cache directory for datasets
        max_train_samples: Maximum training samples (for debugging)
        max_val_samples: Maximum validation samples

    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    task_name = task_name.lower()

    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown GLUE task: {task_name}. Available: {list(GLUE_TASKS.keys())}")

    task_config = GLUE_TASKS[task_name]

    # Create datasets
    train_dataset = GLUEDataset(task_name, 'train', tokenizer, max_length, cache_dir)
    val_dataset = GLUEDataset(task_name, 'validation', tokenizer, max_length, cache_dir)

    # Test set (usually without labels for GLUE)
    try:
        test_dataset = GLUEDataset(task_name, 'test', tokenizer, max_length, cache_dir)
    except:
        print(f"Warning: Test set not available for {task_name}, using validation as test")
        test_dataset = val_dataset

    # Optionally limit samples
    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset.data = train_dataset.data.select(range(max_train_samples))
    if max_val_samples and len(val_dataset) > max_val_samples:
        val_dataset.data = val_dataset.data.select(range(max_val_samples))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get vocabulary size
    if isinstance(train_dataset.tokenizer, SimpleTokenizer):
        vocab_size = len(train_dataset.tokenizer.word2idx)
    else:
        vocab_size = train_dataset.tokenizer.vocab_size

    metadata = {
        'task_name': task_name,
        'task_config': task_config,
        'num_labels': task_config['num_labels'],
        'is_regression': task_config.get('is_regression', False),
        'metric': task_config['metric'],
        'vocab_size': vocab_size,
        'max_length': max_length,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }

    return train_loader, val_loader, test_loader, metadata


# ================================================================================
# Utility Functions
# ================================================================================

def get_glue_task_info(task_name: str) -> Dict:
    """Get information about a GLUE task."""
    task_name = task_name.lower()
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(GLUE_TASKS.keys())}")
    return GLUE_TASKS[task_name]


def list_glue_tasks() -> List[str]:
    """List all available GLUE tasks."""
    return list(GLUE_TASKS.keys())


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GLUE Benchmark Data Loader - Testing")
    print("=" * 80)

    # Test with SST-2 (simple single-sentence task)
    print("\n[1] Testing SST-2 loading...")

    try:
        train_loader, val_loader, test_loader, metadata = load_glue_task(
            task_name='sst2',
            max_length=64,
            batch_size=8,
            num_workers=0,
            max_train_samples=100,  # Limit for testing
            max_val_samples=50
        )

        print(f"  Task: {metadata['task_name']}")
        print(f"  Num labels: {metadata['num_labels']}")
        print(f"  Vocab size: {metadata['vocab_size']}")
        print(f"  Train size: {metadata['train_size']}")
        print(f"  Val size: {metadata['val_size']}")

        # Get a batch
        batch = next(iter(train_loader))
        print(f"\n  Batch shapes:")
        print(f"    input_ids: {batch['input_ids'].shape}")
        print(f"    attention_mask: {batch['attention_mask'].shape}")
        print(f"    labels: {batch['labels'].shape}")

    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure 'datasets' and 'transformers' are installed:")
        print("    pip install datasets transformers")

    # Test MRPC (sentence pair task)
    print("\n[2] Testing MRPC (sentence pair) loading...")

    try:
        train_loader, val_loader, test_loader, metadata = load_glue_task(
            task_name='mrpc',
            max_length=128,
            batch_size=8,
            num_workers=0,
            max_train_samples=100
        )

        print(f"  Task: {metadata['task_name']}")
        print(f"  Train size: {metadata['train_size']}")

        batch = next(iter(train_loader))
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")

    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("Available GLUE Tasks:")
    for task, config in GLUE_TASKS.items():
        print(f"  {task}: {config['name']} ({config['type']}, {config['num_labels']} labels)")
    print("=" * 80)
