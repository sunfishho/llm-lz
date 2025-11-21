"""Data loading and preprocessing utilities for LLM-guided LZ78 learning."""

import os
import pickle
import random
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import glob
from data_classes import BitVector, BitSequence, Sequence, EncodedSequence


class TextDataset(Dataset):
    """Dataset for loading and processing text data."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        min_length: int = 50,
        alphabet_size: int = 256
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.alphabet_size = alphabet_size
        
        # Filter texts by length
        self.texts = [text for text in self.texts if len(text) >= min_length]
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        # Truncate if too long
        if len(text) > self.max_length:
            start = random.randint(0, len(text) - self.max_length)
            text = text[start:start + self.max_length]
        
        # Convert to bytes for LZ78 (byte-level processing)
        text_bytes = text.encode('utf-8')
        
        # Truncate to fit in our max_length
        if len(text_bytes) > self.max_length:
            text_bytes = text_bytes[:self.max_length]
        
        return {
            'text': text,
            'text_bytes': text_bytes,
            'length': len(text_bytes)
        }


class LogitsCache:
    """Cache for LLM logits to avoid recomputation."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, text_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get cached logits for text."""
        text_hash = str(hash(text))
        cache_path = self._get_cache_path(text_hash)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, text: str, logits: torch.Tensor) -> None:
        """Cache logits for text."""
        text_hash = str(hash(text))
        cache_path = self._get_cache_path(text_hash)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(logits, f)


def load_text_data(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    data_dir: str = "./data",
    split: str = "train",
    max_samples: Optional[int] = None
) -> List[str]:
    """Load text data from various sources."""
    
    if dataset_name is not None:
        # Load from HuggingFace datasets
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Extract text field (common field names)
        text_field = None
        for field in ['text', 'content', 'article', 'sentence']:
            if field in dataset.features:
                text_field = field
                break
        
        if text_field is None:
            raise ValueError(f"Could not find text field in dataset {dataset_name}")
        
        texts = [item[text_field] for item in dataset]
        
    else:
        # Load from local files
        text_files = []
        for ext in ['*.txt', '*.json', '*.jsonl']:
            text_files.extend(glob.glob(os.path.join(data_dir, ext)))
        
        texts = []
        for file_path in text_files:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if 'text' in data:
                            texts.append(data['text'])
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                texts.append(item)
                            elif isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
            else:  # .txt
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    
    if max_samples:
        texts = texts[:max_samples]
    
    return texts


def create_data_splits(
    texts: List[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split texts into train/val/test sets."""
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    random.seed(seed)
    texts = texts.copy()
    random.shuffle(texts)
    
    n_total = len(texts)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]
    
    return train_texts, val_texts, test_texts


def create_dataloaders(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    min_length: int = 50,
    alphabet_size: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test sets."""
    
    train_dataset = TextDataset(
        train_texts, tokenizer, max_length, min_length, alphabet_size
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, max_length, min_length, alphabet_size
    )
    test_dataset = TextDataset(
        test_texts, tokenizer, max_length, min_length, alphabet_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batching."""
    texts = [item['text'] for item in batch]
    text_bytes = [item['text_bytes'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    return {
        'texts': texts,
        'text_bytes': text_bytes,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


def bytes_to_lz78_sequence(text_bytes: bytes, alphabet_size: int = 256) -> List[int]:
    """Convert text bytes to LZ78-compatible integer sequence."""
    # Simple modulo mapping to fit within alphabet_size
    return [b % alphabet_size for b in text_bytes]


def lz78_sequence_to_bytes(sequence: List[int], alphabet_size: int = 256) -> bytes:
    """Convert LZ78 sequence back to bytes (for debugging)."""
    # This is lossy if alphabet_size < 256
    return bytes([s % 256 for s in sequence])


def convert_text_to_ascii_bits(text: str):
    ascii_bits = []
    for c in text:
        ascii_val = ord(c)
        for i in range(8):
            # Take bits from most-significant to least-significant
            bit = (ascii_val >> (7 - i)) & 1
            ascii_bits.append(bit)
    return ascii_bits

def _int_to_msb_bits(value: int) -> List[int]:
    """Return the binary digits of value from MSB to LSB."""
    if value <= 0:
        raise ValueError("value must be positive when converting to bits")
    bit_length = value.bit_length()
    return [(value >> shift) & 1 for shift in range(bit_length - 1, -1, -1)]


def elias_omega_coding(value: int, dest: Optional[BitVector] = None) -> BitVector:
    """Append the Elias omega code for value to dest and return dest."""
    if value < 1:
        raise ValueError("Elias omega coding is only defined for positive integers")
    if dest is None:
        dest = BitVector()

    if value == 1:
        dest.push(0, 1)
        return dest

    segments = []
    current = value
    while current > 1:
        segments.append(current)
        current = current.bit_length() - 1

    for segment in reversed(segments):
        for bit in _int_to_msb_bits(segment):
            dest.push(bit, 1)

    dest.push(0, 1)
    return dest


def elias_omega_decoding(data: BitVector, start: int = 0) -> List[int]:
    """Decode every Elias omega code stored in `data` starting at bit `start`."""
    if start < 0:
        raise ValueError("start must be non-negative")
    if start > len(data):
        raise ValueError("start offset beyond available data")

    idx = start
    decoded_values: List[int] = []

    while idx < len(data):
        value, used = _decode_single_elias_omega(data, idx)
        decoded_values.append(value)
        idx += used

    return decoded_values


def _decode_single_elias_omega(data: BitVector, start: int) -> Tuple[int, int]:
    """Decode a single Elias omega code from `data` at bit position `start`."""
    if start >= len(data):
        raise ValueError("start offset beyond available data")

    idx = start
    total_len = len(data)
    value = 1

    while True:
        if idx >= total_len:
            raise ValueError("incomplete Elias omega code: missing terminator bit")
        bit = data.get(idx, 1)
        idx += 1
        if bit == 0:
            return value, idx - start

        if idx + value > total_len:
            raise ValueError("incomplete Elias omega code: missing payload bits")

        decoded = 1
        for _ in range(value):
            decoded = (decoded << 1) | data.get(idx, 1)
            idx += 1
        value = decoded
