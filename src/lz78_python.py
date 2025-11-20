#!/usr/bin/env python3
"""
Python Implementation of LZ78 Compression Algorithm

This is a direct Python translation of the Rust LZ78 encoder implementation.
It includes the core LZ78 algorithm, sequence handling, and compression utilities.
"""

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator, Union, Callable

class BitVector:
    """Wraps Python's integer bit operations to mimic a bit vector."""

    def __init__(self, value: int = 0, bit_length: int = 0):
        self._value = value
        self._length = bit_length

    def push(self, val: int, bitwidth: int):
        """Append the lower bitwidth bits from val, treating bit 0 as the least significant."""
        if bitwidth < 0:
            raise ValueError("bitwidth must be non-negative")
        if bitwidth == 0:
            return
        mask = (1 << bitwidth) - 1
        self._value |= (val & mask) << self._length
        self._length += bitwidth

    def get(self, start: int, length: int) -> int:
        """Return integer composed of [start, start+length) bits."""
        if start < 0 or length < 0:
            raise ValueError("start and length must be non-negative")
        if length == 0:
            return 0
        mask = (1 << length) - 1
        return (self._value >> start) & mask

    def __len__(self):
        return self._length

    def to_bytes(self) -> bytes:
        """Convert to little-endian byte representation."""
        if self._length == 0:
            return b""
        byte_length = (self._length + 7) // 8
        return self._value.to_bytes(byte_length, byteorder="little")

    def from_bytes(self, data: bytes, bit_length: int):
        """Populate from little-endian bytes keeping only bit_length bits."""
        if bit_length < 0:
            raise ValueError("bit_length must be non-negative")
        self._value = int.from_bytes(data, byteorder="little")
        if bit_length < len(data) * 8:
            mask = (1 << bit_length) - 1 if bit_length > 0 else 0
            self._value &= mask
        self._length = bit_length


class LZ78TraversalResult:
    """Result of traversing the LZ78 tree."""
    
    def __init__(self, added_leaf: Optional[int], state_idx: int):
        self.added_leaf = added_leaf
        self.state_idx = state_idx


@dataclass
class LZ78TrieState:
    """Container for an LZ78 trie."""

    transitions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    next_index: Optional[int] = None
    node_counts: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure we keep our own copy of the mapping
        self.transitions = dict(self.transitions)
        # Ensure we keep our own copy of node_counts
        self.node_counts = dict(self.node_counts)
        if self.next_index is None:
            self.next_index = max(self.transitions.values(), default=0) + 1

    def copy(self) -> "LZ78TrieState":
        """Return a deep copy of this state."""
        return LZ78TrieState(self.transitions, self.next_index, self.node_counts)


class LZWData:
    """LZ78 tree data structure for compression."""
    
    def __init__(self, initial_state: Optional[LZ78TrieState] = None):
        if initial_state:
            self.map: Dict[Tuple[int, int], int] = dict(initial_state.transitions)
            self.next_index: int = initial_state.next_index or 1
            self.node_counts: Dict[int, int] = dict(initial_state.node_counts)
        else:
            self.map = {}
            self.next_index = 1
            self.node_counts: Dict[int, int] = {}
    
    def increment_count(self, node_idx: int):
        """Increment the count for a node."""
        self.node_counts[node_idx] = self.node_counts.get(node_idx, 0) + 1
    
    def traverse_root_to_leaf(self, input_iter: Iterator[int]) -> LZ78TraversalResult:
        """Traverse from root to leaf in the LZ78 tree."""
        return self.traverse_to_leaf_from(0, input_iter)
    
    def traverse_to_leaf_from(self, node_idx: int, input_iter: Iterator[int]) -> LZ78TraversalResult:
        """Traverse from a given node to leaf in the LZ78 tree."""
        state_idx = node_idx
        added_leaf = None
        
        # Increment count for starting node
        self.increment_count(state_idx)
        
        for sym in input_iter:
            if (state_idx, sym) in self.map:
                state_idx = self.map[(state_idx, sym)]
                # Increment count for each node we traverse
                self.increment_count(state_idx)
            else:
                # Add new node to tree
                self.map[(state_idx, sym)] = self.next_index
                # Initialize count for new node
                self.node_counts[self.next_index] = 1
                self.next_index += 1
                added_leaf = sym
                break
        
        return LZ78TraversalResult(added_leaf, state_idx)


class EncodedSequence:
    """Stores an encoded bitstream with metadata."""
    
    def __init__(self, data: BitVector, uncompressed_length: int, alphabet_size: int):
        self.data = data
        self.uncompressed_length = uncompressed_length
        self.alphabet_size = alphabet_size
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.uncompressed_length == 0:
            return 0.0
        return len(self.data) / (self.uncompressed_length * math.log2(self.alphabet_size))
    
    def compressed_len_bytes(self) -> int:
        """Length of compressed data in bytes."""
        return (len(self.data) + 7) // 8
    
    def to_bytes(self) -> bytes:
        """Convert to bytes for storage."""
        result = bytearray()
        # Header: alphabet_size (4 bytes), uncompressed_length (8 bytes), data_length (8 bytes)
        result.extend(struct.pack('<I', self.alphabet_size))
        result.extend(struct.pack('<Q', self.uncompressed_length))
        result.extend(struct.pack('<Q', len(self.data)))
        # Data
        result.extend(self.data.to_bytes())
        return bytes(result)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'EncodedSequence':
        """Create from bytes."""
        offset = 0
        alphabet_size = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        uncompressed_length = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        data_length = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        bit_vector = BitVector()
        bit_vector.from_bytes(data[offset:], data_length)
        
        return cls(bit_vector, uncompressed_length, alphabet_size)


class Sequence:
    """Base class for sequences that can be compressed."""
    
    def alphabet_size(self) -> int:
        raise NotImplementedError
    
    def length(self) -> int:
        raise NotImplementedError
    
    def get(self, i: int) -> int:
        raise NotImplementedError
    
    def put_sym(self, sym: int):
        raise NotImplementedError
    
    def iter(self) -> Iterator[int]:
        """Iterator over sequence symbols."""
        for i in range(self.length()):
            yield self.get(i)


class U8Sequence(Sequence):
    """U8 sequence for alphabet sizes 3-256."""
    
    def __init__(self, data: List[int], alphabet_size: int):
        self.data = data
        self.alphabet_size_val = alphabet_size
    
    def alphabet_size(self) -> int:
        return self.alphabet_size_val
    
    def length(self) -> int:
        return len(self.data)
    
    def get(self, i: int) -> int:
        if i >= self.length():
            raise IndexError(f"Index {i} out of range for sequence of length {self.length()}")
        return self.data[i]
    
    def put_sym(self, sym: int):
        if sym >= self.alphabet_size():
            raise ValueError(f"Symbol {sym} not in alphabet of size {self.alphabet_size()}")
        self.data.append(sym)
    
    @classmethod
    def from_data(cls, data: List[int], alphabet_size: int) -> 'U8Sequence':
        """Create from data with validation."""
        if any(x >= alphabet_size for x in data):
            raise ValueError(f"Invalid symbol found for alphabet size of {alphabet_size}")
        return cls(data, alphabet_size)


class CharacterMap:
    """Maps characters to integer symbols."""
    
    def __init__(self, char_to_sym: Dict[str, int], sym_to_char: List[str]):
        self.char_to_sym = char_to_sym
        self.sym_to_char = sym_to_char
        self.alphabet_size = len(sym_to_char)
    
    @classmethod
    def from_data(cls, data: str) -> 'CharacterMap':
        """Create character map from string data."""
        char_to_sym = {}
        sym_to_char = []
        seen_chars = set()
        
        for char in data:
            if char not in seen_chars:
                char_to_sym[char] = len(sym_to_char)
                sym_to_char.append(char)
                seen_chars.add(char)
        
        return cls(char_to_sym, sym_to_char)
    
    def encode(self, char: str) -> Optional[int]:
        """Encode character to symbol."""
        return self.char_to_sym.get(char)
    
    def decode(self, sym: int) -> Optional[str]:
        """Decode symbol to character."""
        if 0 <= sym < self.alphabet_size:
            return self.sym_to_char[sym]
        return None
    
    def encode_all(self, data: str) -> List[int]:
        """Encode entire string."""
        result = []
        for char in data:
            if char in self.char_to_sym:
                result.append(self.char_to_sym[char])
            else:
                raise ValueError(f"Character '{char}' not in mapping")
        return result


class CharacterSequence(Sequence):
    """Character sequence with character mapping."""
    
    def __init__(self, data: str, character_map: CharacterMap):
        self.data = data
        self.character_map = character_map
        self.encoded = character_map.encode_all(data)
    
    def alphabet_size(self) -> int:
        return self.character_map.alphabet_size
    
    def length(self) -> int:
        return len(self.encoded)
    
    def get(self, i: int) -> int:
        if i >= self.length():
            raise IndexError(f"Index {i} out of range for sequence of length {self.length()}")
        return self.encoded[i]
    
    def put_sym(self, sym: int):
        char = self.character_map.decode(sym)
        if char is None:
            raise ValueError(f"Symbol {sym} not in character map")
        self.data += char
        self.encoded.append(sym)
    
    @classmethod
    def from_data(cls, data: str, character_map: CharacterMap) -> 'CharacterSequence':
        """Create from data with character map."""
        return cls(data, character_map)


def lz78_bits_to_encode_phrase(phrase_idx: int, alpha_size: int) -> int:
    """Calculate number of bits needed to encode a phrase."""
    return int(math.ceil(math.log2(phrase_idx + 1) + math.log2(alpha_size)))


def lz78_encode(
    sequence: Sequence,
    capture_tree: bool = False,
    initial_state: Optional[LZ78TrieState] = None,
):
    """Compress a sequence using LZ78.

    Args:
        sequence: Input sequence to encode.
        capture_tree: When True, return a tuple of (EncodedSequence, trie_state).
        initial_state: Optional pre-existing trie state.

    Returns:
        EncodedSequence when capture_tree is False.
        Tuple[EncodedSequence, LZ78TrieState] when capture_tree is True.
    """
    lzw = LZWData(initial_state)
    bits = BitVector()
    
    # Convert to list for easier indexing
    input_data = list(sequence.iter())
    input_idx = 0
    phrase_num = 0
    
    while input_idx < len(input_data):
        # Find the longest match starting from current position
        state_idx = 0
        match_length = 0
        
        # Increment count for root node (state 0) at start of each phrase
        lzw.increment_count(state_idx)
        
        # Traverse the tree to find the longest match
        for i in range(input_idx, len(input_data)):
            sym = input_data[i]
            if (state_idx, sym) in lzw.map:
                state_idx = lzw.map[(state_idx, sym)]
                match_length += 1
                # Increment count for each node we traverse
                lzw.increment_count(state_idx)
            else:
                # Add new node to tree
                lzw.map[(state_idx, sym)] = lzw.next_index
                # Initialize count for new node
                lzw.node_counts[lzw.next_index] = 1
                lzw.next_index += 1
                break
        
        # Calculate bitwidth and encode
        bitwidth = lz78_bits_to_encode_phrase(phrase_num, sequence.alphabet_size())
        phrase_num += 1
        
        # The value to encode: state_idx * alphabet_size + new_symbol
        if input_idx + match_length < len(input_data):
            # There's a new symbol
            new_sym = input_data[input_idx + match_length]
            val = state_idx * sequence.alphabet_size() + new_sym
            input_idx += match_length + 1
        else:
            # End of input - we need to encode the final symbol
            # This happens when we've matched all remaining symbols
            # We need to encode a single symbol that wasn't matched
            if input_idx < len(input_data):
                new_sym = input_data[input_idx]
                val = state_idx * sequence.alphabet_size() + new_sym
                input_idx += 1
            else:
                # No more symbols to process
                break
        
        # Store in bit vector
        bits.push(val, bitwidth)
    
    encoded = EncodedSequence(bits, sequence.length(), sequence.alphabet_size())
    
    if capture_tree:
        return encoded, LZ78TrieState(lzw.map, lzw.next_index, lzw.node_counts)
    return encoded


def lz78_decode(sequence: Sequence, encoded: EncodedSequence) -> None:
    """Decode a sequence using LZ78."""
    # Phrase starts and lengths
    phrase_starts = [0]
    phrase_lengths = [0]
    
    bits_decoded = 0
    alphabet_size = encoded.alphabet_size
    
    while bits_decoded < len(encoded.data):
        # Calculate bitwidth for current phrase
        bitwidth = lz78_bits_to_encode_phrase(len(phrase_starts) - 1, alphabet_size)
        
        # Decode value
        decoded_val = encoded.data.get(bits_decoded, bitwidth)
        bits_decoded += bitwidth
        
        # Find reference index and new symbol
        ref_idx = decoded_val // alphabet_size
        new_sym = decoded_val % alphabet_size
        
        # Get phrase start and length
        phrase_start = sequence.length()
        phrase_len = phrase_lengths[ref_idx] + 1
        copy_start = phrase_starts[ref_idx]
        
        # Copy previous phrase
        for j in range(phrase_len - 1):
            if copy_start + j < sequence.length():
                sequence.put_sym(sequence.get(copy_start + j))
                if sequence.length() >= encoded.uncompressed_length:
                    return
        
        # Add new symbol
        sequence.put_sym(new_sym)
        phrase_lengths.append(phrase_len)
        phrase_starts.append(phrase_start)
        
        # Check if we've decoded enough
        if sequence.length() >= encoded.uncompressed_length:
            break


class LZ78Encoder:
    """LZ78 encoder implementation with optional pretrained trie support."""
    
    def __init__(
        self,
        base_tree: Optional[Union[LZ78TrieState, Dict[Tuple[int, int], int]]] = None,
    ):
        if isinstance(base_tree, LZ78TrieState):
            self.base_tree: Optional[LZ78TrieState] = base_tree.copy()
        elif base_tree:
            # If base_tree is a dict, create LZ78TrieState with empty counts
            self.base_tree = LZ78TrieState(base_tree, None, {})
        else:
            self.base_tree = None
    
    def _select_tree(self, override: Optional[LZ78TrieState]):
        if override is not None:
            return override
        return self.base_tree.copy() if self.base_tree else None
    
    def encode(
        self,
        sequence: Sequence,
        initial_state: Optional[LZ78TrieState] = None,
    ) -> EncodedSequence:
        """Encode a sequence using LZ78, optionally seeding with an existing trie state."""
        tree = self._select_tree(initial_state)
        return lz78_encode(sequence, initial_state=tree)
    
    def encode_with_tree(
        self,
        sequence: Sequence,
        initial_state: Optional[LZ78TrieState] = None,
    ):
        """Encode a sequence and capture the resulting LZ78 trie state."""
        tree = self._select_tree(initial_state)
        return lz78_encode(sequence, capture_tree=True, initial_state=tree)
    
    def pretrain(
        self,
        sequence: Sequence,
        initial_state: Optional[LZ78TrieState] = None,
    ) -> LZ78TrieState:
        """Update the base trie by encoding a sequence (e.g., for warm-starting)."""
        _, tree = self.encode_with_tree(sequence, initial_state=initial_state)
        self.base_tree = tree.copy()
        return self.base_tree.copy()
    
    def get_base_tree(self) -> Optional[LZ78TrieState]:
        """Return a copy of the current base trie state."""
        return self.base_tree.copy() if self.base_tree else None
    
    def tree_state_from_encoded(self, encoded_state: LZ78TrieState) -> LZ78TrieState:
        """Return a copy of an externally provided trie state."""
        return encoded_state.copy()
    
    def decode(self, sequence: Sequence, encoded: EncodedSequence) -> None:
        """Decode a sequence using LZ78."""
        lz78_decode(sequence, encoded)
    
    def visualize_tree(
        self,
        tree_state: LZ78TrieState,
        symbol_decoder: Optional[Callable[[int], Optional[str]]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Return a Graphviz DOT representation of the provided trie state.

        Args:
            tree_state: LZ78TrieState instance to visualize.
            symbol_decoder: Optional callable mapping symbol ints to readable labels.
            output_path: Optional path to write the DOT text.

        Returns:
            DOT source string.
        """
        dot_text = lz78_tree_to_dot(tree_state, symbol_decoder=symbol_decoder)
        if output_path:
            Path(output_path).write_text(dot_text, encoding="utf-8")
        return dot_text


def compress_text(text: str, alphabet_size: int = 256) -> EncodedSequence:
    """Compress text using LZ78."""
    # Convert text to bytes and then to symbols
    text_bytes = text.encode('utf-8')
    symbols = [b % alphabet_size for b in text_bytes]
    
    # Create sequence
    sequence = U8Sequence(symbols, alphabet_size)
    
    # Encode
    encoder = LZ78Encoder()
    return encoder.encode(sequence)


def decompress_with_charmap(encoded: EncodedSequence, character_map: CharacterMap) -> str:
    """Decompress using LZ78 and map symbols back to characters via CharacterMap."""
    # Create an empty character sequence that appends decoded symbols as characters
    sequence = CharacterSequence.from_data("", character_map)
    encoder = LZ78Encoder()
    encoder.decode(sequence, encoded)
    return sequence.data

def _default_symbol_decoder(sym: int) -> str:
    """Fallback label for trie edges."""
    return str(sym)


def _escape_label(label: str) -> str:
    """Escape double quotes for DOT output."""
    return label.replace('"', r'\"')


def lz78_tree_to_dot(
    tree_state: LZ78TrieState,
    symbol_decoder: Optional[Callable[[int], Optional[str]]] = None,
) -> str:
    """Convert an LZ78 trie state into Graphviz DOT format."""
    decoder = symbol_decoder or _default_symbol_decoder
    nodes = {0}
    edges = []
    for (parent, symbol), child in tree_state.transitions.items():
        nodes.add(parent)
        nodes.add(child)
        decoded = decoder(symbol)
        label = str(symbol) if decoded is None else str(decoded)
        edges.append((parent, child, _escape_label(label)))
    
    lines = ["digraph LZ78Trie {", "  rankdir=LR;", '  node [shape=circle, fontsize=10];']
    for node in sorted(nodes):
        lines.append(f'  \"{node}\" [label=\"{node}\"];')
    for parent, child, label in edges:
        lines.append(f'  \"{parent}\" -> \"{child}\" [label=\"{label}\"];')
    lines.append("}")
    return "\n".join(lines)


def main():
    """Test the LZ78 implementation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Python LZ78 encoder")
    parser.add_argument("--text", type=str, help="Text to compress")
    parser.add_argument("--alphabet-size", type=int, default=256, help="Alphabet size")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()
    
    if args.test:
        # Run basic tests
        print("Running LZ78 tests...")
        
        # Test 1: Simple text compression
        test_text = "hello world hello world"
        print(f"Original text: {test_text}")
        
        encoded = compress_text(test_text, 256)
        print(f"Compressed size: {encoded.compressed_len_bytes()} bytes")
        print(f"Compression ratio: {encoded.compression_ratio():.4f}")
        
        decompressed = decompress_with_charmap(encoded, charmap)
        print(f"Decompressed text: {decompressed}")
        print(f"Lossless: {test_text == decompressed}")
        
        # Test 2: Different alphabet sizes
        print("\nTesting different alphabet sizes:")
        for alpha_size in [8, 64, 256]:
            encoded = compress_text(test_text, alpha_size)
            print(f"Alphabet size {alpha_size}: ratio={encoded.compression_ratio():.4f}")
        
    elif args.text:
        # Compress provided text
        print(f"Compressing: {args.text}")
        encoded = compress_text(args.text, args.alphabet_size)
        
        print(f"Original length: {len(args.text)} characters")
        print(f"Compressed size: {encoded.compressed_len_bytes()} bytes")
        print(f"Compression ratio: {encoded.compression_ratio():.4f}")
        
        # Test decompression
        decompressed = decompress_with_charmap(encoded, charmap)
        print(f"Decompressed: {decompressed}")
        print(f"Lossless: {args.text == decompressed}")
    
    else:
        # Default test
        sample_text = "The quick brown fox jumps over the lazy dog. " * 5
        print(f"Compressing sample text: {sample_text[:50]}...")
        
        encoded = compress_text(sample_text, 256)
        print(f"Original length: {len(sample_text)} characters")
        print(f"Compressed size: {encoded.compressed_len_bytes()} bytes")
        print(f"Compression ratio: {encoded.compression_ratio():.4f}")
        
        decompressed = decompress_with_charmap(encoded, charmap)
        print(f"Lossless: {sample_text == decompressed}")


if __name__ == "__main__":
    # Example: visualize trie after encoding
    text = "123531213112"
    charmap = CharacterMap.from_data(text)
    char_seq = CharacterSequence.from_data(text, charmap)
    encoder = LZ78Encoder()
    encoded, trie_state = encoder.encode_with_tree(char_seq)
    dot = encoder.visualize_tree(trie_state, symbol_decoder=charmap.decode, output_path="lz78.dot")
    import subprocess
    subprocess.run(['dot', '-Tpng', 'lz78.dot', '-o', 'lz78.png'], check=True)
    decoded = decompress_with_charmap(encoded, charmap)
    print(f"Decoded text: {decoded}")
