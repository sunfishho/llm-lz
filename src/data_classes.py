from typing import List, Iterator
import math
import struct


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


class BitSequence(Sequence):
    """A sequence of bits to encode with LZ78."""
    def __init__(self, bits: List[int]):
        self.bits = bits
    def alphabet_size(self):
        return 2
    def length(self):
        return len(self.bits)
    def get(self, i):
        return self.bits[i]
    def put_sym(self, sym):
        self.bits.append(sym)
    def iter(self):
        return iter(self.bits)

class EncodedSequence(Sequence):
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