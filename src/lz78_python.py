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
from data_utils import convert_text_to_ascii_bits, elias_omega_coding, elias_omega_decoding, BitVector, BitSequence


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
    node_counts: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure we keep our own copy of the mapping
        self.transitions = dict(self.transitions)
        # Ensure we keep our own copy of node_counts
        self.node_counts = {k: list(v) for k, v in self.node_counts.items()}
        if self.next_index is None:
            self.next_index = max(self.transitions.values(), default=0) + 1

    def copy(self) -> "LZ78TrieState":
        """Return a deep copy of this state."""
        return LZ78TrieState(self.transitions, self.next_index, self.node_counts)


class LZ78Tree:
    """LZ78 tree data structure for compression."""
    
    def __init__(self):
        self.map = {}
        self.next_index = 1
        self.node_counts: Dict[int, List[int]] = {0: [0, 0]}
        self.node_to_init_time: Dict[int, int] = {0: -1}
        self.parent_map: Dict[int, Tuple[int, int]] = {0: None}
    
    def increment_count(self, node_idx: int, symbol: int):
        """Increment the count for traversing a symbol edge from node_idx."""
        if node_idx not in self.node_counts:
            self.node_counts[node_idx] = [0, 0]
        self.node_counts[node_idx][symbol] += 1


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


class LZ78Encoder:
    """LZ78 encoder implementation with optional pretrained trie support."""
    
    def __init__(self):
        self.base_tree = None
        self.encoded_stream: List[Tuple[int, int]] = []
        self.tree = LZ78Tree()
    
    def encode(
            self,
            sequence: Sequence,
            capture_tree: bool = False,
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
        bits = BitVector()
        
        # Convert to list for easier indexing
        input_data = list(sequence.iter())
        input_idx = 0
        phrase_num = 0
        log_likelihood = 0.0
        lz78 = self.tree
        
        while input_idx < len(input_data):
            # Find the longest match starting from current position
            state_idx = 0
            match_length = 0
            terminated = False
            last_sym = None
            # Traverse the tree to find the longest match
            for i in range(input_idx, len(input_data)):
                sym = input_data[i]
                last_sym = sym
                prob = (lz78.node_counts.get(state_idx, [0,0])[sym] + 1) / (sum(lz78.node_counts.get(state_idx, [0,0])) + 2)
                log_likelihood += math.log2(prob)
                lz78.increment_count(state_idx, sym)
                if (state_idx, sym) in lz78.map:
                    # Count this edge traversal (we're traversing FROM state_idx with symbol sym)
                    state_idx = lz78.map[(state_idx, sym)]
                    match_length += 1
                else:
                    # Add new node to tree
                    new_node = lz78.next_index
                    lz78.map[(state_idx, sym)] = new_node
                    lz78.node_counts[new_node] = [0, 0]
                    lz78.next_index += 1
                    lz78.node_to_init_time[new_node] = phrase_num
                    lz78.parent_map[new_node] = state_idx
                    terminated = True
                    break
            
            # Calculate bitwidth and encode
            
            # The value to encode: state_idx * alphabet_size + new_symbol
            if (input_idx + match_length + 1 >= len(input_data)):
                # we keep adding 0s until we hit a leaf node
                num_appended = 0
                while (state_idx, last_sym) in lz78.map:
                    state_idx = lz78.map[(state_idx, last_sym)]
                    num_appended += 1
                if not terminated:
                    # if we need padding, we encode this with num_zeros + 1
                    # if we don't need padding, we encode this with 1
                    num_appended += 1
                if num_appended > 0:
                    val = (phrase_num - lz78.node_to_init_time[lz78.parent_map[state_idx]] - 1) * sequence.alphabet_size() + last_sym
                    elias_omega_coding(val + 1, bits)
                elias_omega_coding(num_appended + 1, bits)
                break
            # There's a new symbol
            new_sym = input_data[input_idx + match_length]
            # Encode the pair (num of phrases since parent node was created, new symbol)
            val = (phrase_num - lz78.node_to_init_time[lz78.parent_map[new_node]] - 1) * sequence.alphabet_size() + new_sym
            input_idx += match_length + 1
            phrase_num += 1
            # Store in bit vector using Elias omega codes
            elias_omega_coding(val + 1, bits)
        
        encoded = EncodedSequence(bits, sequence.length(), sequence.alphabet_size())
        
        if capture_tree:
            return encoded, LZ78TrieState(lz78.map, lz78.next_index, lz78.node_counts), log_likelihood
        return encoded, log_likelihood

    def decode(self, encoded: EncodedSequence) -> BitSequence:
        """Decode a sequence using LZ78."""
        encoded_bits = encoded.data
        decoded_integers = [x - 1 for x in elias_omega_decoding(encoded_bits)]
        padding = decoded_integers[-1]
        decoded_integers = decoded_integers[:len(decoded_integers) - 1]
        tuple_list = []
        output_data: List[int] = []
        for integer in decoded_integers:
            phrase_diff = integer // 2
            symbol = integer % 2
            tuple_list.append((phrase_diff, symbol))
            counter = len(tuple_list) - 1
            symbols_in_phrase = []
            while counter >= 0:
                current_phrase_diff, current_symbol = tuple_list[counter]
                symbols_in_phrase.append(current_symbol)
                counter -= (current_phrase_diff + 1)
            output_data.extend(reversed(symbols_in_phrase))
        if padding > 0:
            # this is checking for the case where we just reached the root node after finishing a phrase
            output_data = output_data[:len(output_data) - padding + 1]
        return BitSequence(output_data)

    def encode_with_tree(
        self,
        sequence: Sequence,
        initial_state: Optional[LZ78TrieState] = None,
    ):
        """Encode a sequence and capture the resulting LZ78 trie state."""
        return self.encode(sequence, capture_tree=True)
    
    def pretrain(
        self,
        sequence: Sequence,
        initial_state: Optional[LZ78TrieState] = None,
    ) -> LZ78TrieState:
        """Update the base trie by encoding a sequence (e.g., for warm-starting)."""
        _, tree = self.encode_with_tree(sequence, initial_state=initial_state)
    
    def get_base_tree(self) -> Optional[LZ78TrieState]:
        """Return a copy of the current base trie state."""
        return self.base_tree.copy() if self.base_tree else None
    
    def tree_state_from_encoded(self, encoded_state: LZ78TrieState) -> LZ78TrieState:
        """Return a copy of an externally provided trie state."""
        return encoded_state.copy()
    
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


def decompress_bits(encoded: EncodedSequence) -> List[int]:
    """Decompress bit sequence using LZ78."""
    # Create an empty bit sequence for decoding
    encoder = LZ78Encoder()
    sequence = encoder.decode(encoded)
    return sequence.bits

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

if __name__ == "__main__":
    # Example: visualize trie after encoding
    # Check that for all binary strings up to length 5, decompression matches original
    from itertools import product

    max_len = 5
    all_ok = True
    for n in range(1, max_len + 1):
        for bits_tuple in product([0,1], repeat=n):
            orig_bits = list(bits_tuple)
            bitseq = BitSequence(orig_bits)
            encoder = LZ78Encoder()
            encoded, _, _ = encoder.encode_with_tree(bitseq)
            decoded = decompress_bits(encoded)
            if decoded != orig_bits:
                print(f"Fail for {orig_bits}: decoded {decoded}")
                all_ok = False
    if all_ok:
        print("All binary strings of length ≤ 5 were encoded/decoded losslessly.")
    # ascii_bits = convert_text_to_ascii_bits(text)
    # print(ascii_bits)
    # bit_sequence = BitSequence(ascii_bits)
    # encoder = LZ78Encoder()
    # encoded_bits, trie_state_bits, log_likelihood = encoder.encode_with_tree(bit_sequence)
    # # Visualize without charmap - just use default symbol decoder (shows 0/1 for bits)
    # dot = encoder.visualize_tree(trie_state_bits, output_path="lz78.dot")
    # import subprocess
    # subprocess.run(['dot', '-Tpng', 'lz78.dot', '-o', 'lz78.png'], check=True)
    # decoded_bits = decompress_bits(encoded_bits)
    # print(decoded_bits)
    # print(f"Match: {ascii_bits == decoded_bits}")
    
    # Print counts at each node
    # print("\nNode counts (node_idx: [count_0, count_1]):")
    # for node_idx in sorted(trie_state_bits.node_counts.keys()):
    #     print(f"  Node {node_idx}: {trie_state_bits.node_counts[node_idx]}")
    # print(f"Log likelihood: {log_likelihood}")
