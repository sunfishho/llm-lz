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
from data_utils import convert_text_to_ascii_bits, elias_omega_coding, elias_omega_decoding
from data_classes import BitVector, BitSequence, Sequence, EncodedSequence
from itertools import product


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


class LZ78Encoder:
    """LZ78 encoder implementation with optional pretrained trie support."""
    
    def __init__(self):
        self.base_tree = None
        self.encoded_stream: List[Tuple[int, int]] = []
        self.tree = LZ78Tree()
    
    def encode(
            self,
            sequence: BitSequence,
            capture_tree: bool = False,
            include_last_subphrase: bool = True,
    ) -> Tuple[EncodedSequence, Optional[LZ78TrieState], float]:
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
        input_data = sequence.bits[:]
        input_idx = 0
        phrase_num = 0
        log_likelihood: float = 0.0
        lz78 = self.tree
        while input_idx < len(input_data):
            # Find the longest match starting from current position
            state_idx = 0
            match_length = 0
            terminated = False
            # Traverse the tree to find the longest match
            for i in range(input_idx, len(input_data)):
                sym = input_data[i]
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
            if terminated:
                # There's a new symbol
                new_sym = input_data[input_idx + match_length]
                # Encode the pair (num of phrases since parent node was created, new symbol)
                val = (phrase_num - lz78.node_to_init_time[lz78.parent_map[new_node]] - 1) * sequence.alphabet_size() + new_sym
                phrase_num += 1
                # Store in bit vector using Elias omega codes
                elias_omega_coding(val + 1, bits)
            if (input_idx + match_length + 1 >= len(input_data)):
                # at the end of the input
                if not include_last_subphrase:
                    # we need to check if we just started a new subphrase or if we are in the middle of a subphrase
                    # do not put padding
                    break
                num_appended = 0
                if not terminated:
                    num_appended += 1
                # we keep adding sym until we hit a leaf node
                while (state_idx, sym) in lz78.map and not terminated:
                    state_idx = lz78.map[(state_idx, sym)]
                    num_appended += 1
                if num_appended > 0:
                    # if terminated is true, we will not reach here so we don't need to worry about state_idx being 0
                    val = (phrase_num - lz78.node_to_init_time[state_idx] - 1) * sequence.alphabet_size() + sym
                    elias_omega_coding(val + 1, bits)
                elias_omega_coding(num_appended + 1, bits)
                break
            if terminated:
                # need to update input_idx but not before we check the condition for hitting the end of the data
                input_idx += match_length + 1
        encoded = EncodedSequence(bits, sequence.length(), sequence.alphabet_size())
        
        if capture_tree:
            return encoded, LZ78TrieState(lz78.map, lz78.next_index, lz78.node_counts), log_likelihood
        return encoded, log_likelihood

    @classmethod
    def decode(cls, encoded: EncodedSequence) -> BitSequence:
        """Decode a sequence using LZ78."""
        encoded_bits = encoded.data
        decoded_integers = [x - 1 for x in elias_omega_decoding(encoded_bits)]
        padding = decoded_integers[-1]
        decoded_integers = decoded_integers[:-1]
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
            output_data = output_data[:len(output_data) - padding]
        return BitSequence(output_data)

    @classmethod
    def find_pretrain_encoded(cls, pretrain_sequence: BitSequence) -> Tuple[EncodedSequence, float]:
        pretrain_encoder = LZ78Encoder()
        pretrain_encoded, pretrain_ll = pretrain_encoder.encode(pretrain_sequence, include_last_subphrase=False)
        return pretrain_encoded, pretrain_ll

    @classmethod
    def pretrain(cls, pretrain_sequence: BitSequence, main_sequence: BitSequence):
        total_sequence = pretrain_sequence + main_sequence
        total_encoder = LZ78Encoder()
        total_encoded, total_ll = total_encoder.encode(total_sequence)
        pretrain_encoded, pretrain_ll = cls.find_pretrain_encoded(pretrain_sequence)
        len_pretrain_encoded = pretrain_encoded.data._length
        encoded_length = total_encoded.uncompressed_length - pretrain_encoded.uncompressed_length
        encoded_bitvector = BitVector(total_encoded.data.get(len_pretrain_encoded), total_encoded.data._length - len_pretrain_encoded)
        return EncodedSequence(encoded_bitvector, encoded_length, total_encoded.alphabet_size), total_ll - pretrain_ll

    @classmethod
    def decode_pretrained(cls, pretrain_sequence: BitSequence, encoded: EncodedSequence) -> BitSequence:
        pretrain_encoded, _ = cls.find_pretrain_encoded(pretrain_sequence)
        total_decoded = cls.decode(pretrain_encoded + encoded)
        len_pretrain = len(pretrain_sequence.bits)
        return BitSequence(total_decoded.bits[len_pretrain:])
    
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

    # bitseq = BitSequence([1,0,1,0,1])
    # encoder = LZ78Encoder()
    # encoded, _ = encoder.encode(bitseq, include_last_subphrase=False)
    # decoded = LZ78Encoder.decode(encoded).bits
    # print(f'decoded: {decoded}')

    # max_len = 10
    # all_ok = True
    # for n in range(1, max_len + 1):
    #     for bits_tuple in product([0,1], repeat=n):
    #         print(f'bits_tuple: {bits_tuple}')
    #         orig_bits = list(bits_tuple)
    #         bitseq = BitSequence(orig_bits)
    #         encoder = LZ78Encoder()
    #         encoded, _ = encoder.encode(bitseq)
    #         decoded = LZ78Encoder.decode(encoded).bits
    #         if decoded != orig_bits:
    #             print(f"Fail for {orig_bits}: decoded {decoded}")
    #             all_ok = False
    # if all_ok:
    #     print(f"All binary strings of length ≤ {max_len} were encoded/decoded losslessly.")

    # bitseq = BitSequence([1])
    # pretrain_bits = BitSequence([1])
    # encoded, ll = LZ78Encoder.pretrain(pretrain_bits, bitseq)
    from itertools import product

    def check_pretrain_encoder_all_binary_strings():
        max_len = 3
        pretrain_len = 5
        all_ok = True

        encoder = LZ78Encoder()

        for data_bits in product([0, 1], repeat=max_len):
            for pre_bits in product([0, 1], repeat=pretrain_len):
                bitseq = BitSequence(list(data_bits))
                pretrain_bits = BitSequence(list(pre_bits))
                encoder = LZ78Encoder()
                encoded, _ = encoder.pretrain(pretrain_bits, bitseq)
                pretrain_encoder = LZ78Encoder()
                pretrain_encoded, _ = pretrain_encoder.encode(pretrain_bits, include_last_subphrase=False)
                total_encoder = LZ78Encoder()
                total_encoded, _ = total_encoder.encode(pretrain_bits + bitseq)
                # print(f'pretrain_encoded: {pretrain_encoded.data._value, pretrain_encoded.data._length}')
                # print(f'encoded: {encoded.data._value, encoded.data._length}')
                # print(f'pretrain + encoded: {total_encoded.data._value, total_encoded.data._length}')

                # Decode
                decoded = LZ78Encoder.decode_pretrained(pretrain_bits, encoded).bits
                if decoded != list(data_bits):
                    print(f"Fail for data={list(data_bits)} with pretrain={list(pre_bits)}: decoded {decoded}")
                    all_ok = False
        if all_ok:
            print(f"All bit strings of length {max_len} compressed with all pretrainings of length {pretrain_len} succeed with lossless roundtrip.")

    check_pretrain_encoder_all_binary_strings()
    
    # Print counts at each node
    # print("\nNode counts (node_idx: [count_0, count_1]):")
    # for node_idx in sorted(trie_state_bits.node_counts.keys()):
    #     print(f"  Node {node_idx}: {trie_state_bits.node_counts[node_idx]}")
    # print(f"Log likelihood: {log_likelihood}")
    pass