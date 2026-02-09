"""
Character-level text dataset loaders for Quantum Latent CFM
=============================================================

Provides a unified interface for loading text corpora as character-level
sequences with a fixed 27-character vocabulary (space + a-z).

Supported datasets:
  - text8         : 100M chars from cleaned Wikipedia (Mahoney, 2011)
  - wikitext-2    : ~2M tokens from Wikipedia Good/Featured articles
  - wikitext-103  : ~103M tokens from Wikipedia Good/Featured articles
  - ptb           : Penn Treebank (~929K train tokens, Mikolov preprocessing)

All loaders return: (train_loader, val_loader, test_loader)
  where each batch is a tuple (tokens,) with tokens: (B, seq_len) int64 ∈ [0,26]

Usage:
    from data.Load_Text_Datasets import load_text_dataset

    train_loader, val_loader, test_loader = load_text_dataset(
        dataset="wikitext-2", seq_len=256, n_train=10000,
        n_valtest=2000, batch_size=64)
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Shared constants — same 27-char vocab across all datasets
# ---------------------------------------------------------------------------
VOCAB_SIZE = 27
CHAR2IDX = {' ': 0}
CHAR2IDX.update({chr(ord('a') + i): i + 1 for i in range(26)})
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}

DATA_ROOT = os.environ.get("TEXT_DATA_ROOT",
                           "/pscratch/sd/j/junghoon/data")


def decode_tokens(token_tensor):
    """Convert int token tensor back to string."""
    return ''.join(IDX2CHAR.get(t.item(), '?') for t in token_tensor)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clean_text(raw_text):
    """Lowercase and keep only chars in CHAR2IDX (space + a-z)."""
    raw_text = raw_text.lower()
    return ''.join(c for c in raw_text if c in CHAR2IDX)


def _text_to_dataloaders(text, dataset_name, seq_len, n_train, n_valtest,
                         batch_size):
    """Convert a cleaned character string into train/val/test DataLoaders."""
    print(f"{dataset_name}: {len(text):,} characters after cleaning")

    data = np.array([CHAR2IDX[c] for c in text], dtype=np.int64)

    total_needed = (n_train + n_valtest) * seq_len
    if total_needed > len(data):
        total_needed = len(data)
        total_chunks = total_needed // seq_len
        n_train = min(n_train, int(total_chunks * 0.8))
        n_valtest = total_chunks - n_train
        print(f"  Adjusted: n_train={n_train}, n_valtest={n_valtest}")

    n_chunks = total_needed // seq_len
    data = data[:n_chunks * seq_len].reshape(n_chunks, seq_len)
    data = torch.from_numpy(data).long()

    perm = torch.randperm(n_chunks)
    data = data[perm]

    X_train = data[:n_train]
    X_valtest = data[n_train:n_train + n_valtest]

    train_ds = TensorDataset(X_train)
    valtest_ds = TensorDataset(X_valtest)
    val_sz = len(valtest_ds) // 2
    test_sz = len(valtest_ds) - val_sz
    val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Splits: train={len(X_train)}  val={val_sz}  test={test_sz}  "
          f"seq_len={seq_len}")

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# text8
# ---------------------------------------------------------------------------
def load_text8(seq_len, n_train, n_valtest, batch_size):
    """Load text8 dataset (100M chars of cleaned Wikipedia).

    Source: http://mattmahoney.net/dc/text8.zip
    Already lowercase a-z + space, so no cleaning needed.
    """
    cache_dir = os.path.join(DATA_ROOT, "text8")
    os.makedirs(cache_dir, exist_ok=True)
    text8_path = os.path.join(cache_dir, "text8")

    if not os.path.exists(text8_path):
        zip_path = os.path.join(cache_dir, "text8.zip")
        url = "http://mattmahoney.net/dc/text8.zip"
        print(f"Downloading text8 from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)
        os.remove(zip_path)
        print(f"Extracted to {text8_path}")

    with open(text8_path, 'r') as f:
        text = f.read().strip()

    text = ''.join(c for c in text if c in CHAR2IDX)
    return _text_to_dataloaders(text, "text8", seq_len, n_train, n_valtest,
                                batch_size)


# ---------------------------------------------------------------------------
# WikiText-2 / WikiText-103
# ---------------------------------------------------------------------------
def _load_wikitext_hf(variant, seq_len, n_train, n_valtest, batch_size):
    """Load WikiText-2 or WikiText-103 via HuggingFace datasets.

    WikiText articles are concatenated into a single character stream,
    lowercased, and filtered to the 27-char vocab.

    Args:
        variant: "wikitext-2-raw-v1" or "wikitext-103-raw-v1"
    """
    from datasets import load_dataset

    dataset_name = variant.replace("-raw-v1", "").replace("-", " ").title()

    cache_dir = os.path.join(DATA_ROOT, "wikitext")
    os.makedirs(cache_dir, exist_ok=True)

    # Check for cached cleaned text
    clean_name = variant.replace("-raw-v1", "").replace("-", "_")
    cache_path = os.path.join(cache_dir, f"{clean_name}_clean.txt")

    if os.path.exists(cache_path):
        print(f"Loading cached {dataset_name} from {cache_path}")
        with open(cache_path, 'r') as f:
            text = f.read()
    else:
        print(f"Downloading {dataset_name} via HuggingFace datasets...")
        ds = load_dataset("wikitext", variant, cache_dir=cache_dir)

        # Concatenate all splits for maximum data, then we split ourselves
        parts = []
        for split in ["train", "validation", "test"]:
            lines = ds[split]["text"]
            parts.append('\n'.join(lines))
        raw = '\n'.join(parts)

        text = _clean_text(raw)

        # Cache cleaned text
        with open(cache_path, 'w') as f:
            f.write(text)
        print(f"Cached cleaned text to {cache_path}")

    return _text_to_dataloaders(text, dataset_name, seq_len, n_train,
                                n_valtest, batch_size)


def load_wikitext2(seq_len, n_train, n_valtest, batch_size):
    """Load WikiText-2 (~2M tokens, ~10M chars)."""
    return _load_wikitext_hf("wikitext-2-raw-v1", seq_len, n_train,
                             n_valtest, batch_size)


def load_wikitext103(seq_len, n_train, n_valtest, batch_size):
    """Load WikiText-103 (~103M tokens, ~500M chars)."""
    return _load_wikitext_hf("wikitext-103-raw-v1", seq_len, n_train,
                             n_valtest, batch_size)


# ---------------------------------------------------------------------------
# Penn Treebank (PTB)
# ---------------------------------------------------------------------------
def load_ptb(seq_len, n_train, n_valtest, batch_size):
    """Load Penn Treebank (Mikolov preprocessing).

    Downloads from the original Zaremba et al. (2014) GitHub repository.
    PTB text is concatenated, lowercased, and filtered to 27-char vocab.
    The <unk> token and newlines are replaced with spaces.

    Source: https://github.com/wojzaremba/lstm/tree/master/data
    """
    cache_dir = os.path.join(DATA_ROOT, "ptb")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, "ptb_clean.txt")

    if os.path.exists(cache_path):
        print(f"Loading cached PTB from {cache_path}")
        with open(cache_path, 'r') as f:
            text = f.read()
    else:
        base_url = ("https://raw.githubusercontent.com/"
                    "wojzaremba/lstm/master/data/")
        parts = []
        for split in ["train", "valid", "test"]:
            fname = f"ptb.{split}.txt"
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                url = base_url + fname
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, fpath)
            with open(fpath, 'r') as f:
                parts.append(f.read())

        raw = ' '.join(parts)
        raw = raw.replace('<unk>', ' ')
        raw = raw.replace('\n', ' ')

        text = _clean_text(raw)

        with open(cache_path, 'w') as f:
            f.write(text)
        print(f"Cached cleaned text to {cache_path}")

    return _text_to_dataloaders(text, "Penn Treebank", seq_len, n_train,
                                n_valtest, batch_size)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "text8": load_text8,
    "wikitext-2": load_wikitext2,
    "wikitext-103": load_wikitext103,
    "ptb": load_ptb,
}


def load_text_dataset(dataset, seq_len, n_train, n_valtest, batch_size):
    """Load a character-level text dataset by name.

    Args:
        dataset: One of "text8", "wikitext-2", "wikitext-103", "ptb"
        seq_len: Character sequence length per sample
        n_train: Number of training sequences
        n_valtest: Number of validation+test sequences (split 50/50)
        batch_size: Batch size for DataLoaders

    Returns:
        (train_loader, val_loader, test_loader)
        Each batch: (tokens,) with tokens (B, seq_len) int64 in [0, 26]
    """
    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset](seq_len, n_train, n_valtest, batch_size)


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test text dataset loaders")
    p.add_argument("--dataset", type=str, default="text8",
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-valtest", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    train_ld, val_ld, test_ld = load_text_dataset(
        args.dataset, args.seq_len, args.n_train, args.n_valtest,
        args.batch_size)

    batch = next(iter(train_ld))
    tokens = batch[0]
    print(f"\nBatch shape: {tokens.shape}  dtype: {tokens.dtype}")
    print(f"Value range: [{tokens.min().item()}, {tokens.max().item()}]")
    print(f"\nSample 0: '{decode_tokens(tokens[0][:80])}'...")
    print(f"Sample 1: '{decode_tokens(tokens[1][:80])}'...")
