from typing import List, Iterator
import math
import struct


class BitVector:
    """Wraps Python's integer bit operations to mimic a bit vector."""

    def __init__(self, value: int = 0, bit_length: int | None = None):
        self._value = value
        if bit_length is None:
            bit_length = math.ceil(math.log2(value + 1))
        self._length = bit_length

    def push(self, val: int, bitwidth: int | None = None):
        """Append the lower bitwidth bits from val, treating bit 0 as the least significant."""
        if bitwidth is None:
            bitwidth = math.ceil(math.log2(val + 1))
        if bitwidth < 0:
            raise ValueError("bitwidth must be non-negative")
        if bitwidth == 0:
            return
        mask = (1 << bitwidth) - 1
        self._value |= (val & mask) << self._length
        self._length += bitwidth

    def get(self, start: int, length: int | None = None) -> int:
        """Return integer composed of [start, start+length) bits."""
        if length is None:
            length = self._length - start
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

    def __add__(self, other: 'BitVector') -> 'BitVector':
        return BitVector(self._value + (other._value << self._length), self._length + other._length)


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

    def __add__(self, other: 'Sequence') -> 'Sequence':
        raise NotImplementedError

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
    def __add__(self, other: 'BitSequence') -> 'BitSequence':
        if not isinstance(other, BitSequence):
            return NotImplemented
        return BitSequence(self.bits + other.bits)

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
    
    def __add__(self, other: 'EncodedSequence') -> 'EncodedSequence':
        return EncodedSequence(self.data + other.data, self.uncompressed_length + other.uncompressed_length, self.alphabet_size)

if __name__ == "__main__":
    def test_bitvector_addition():
        # Test BitVector addition (__add__)
        bv1 = BitVector(11)
        bv2 = BitVector(1, 3)
        bv_sum = bv1 + bv2
        assert bv_sum._value == 27 and bv_sum._length == 7, f"BitVector add failed: {bv_sum._value}, {bv_sum._length}"

    def test_bitsequence_addition():
        # Test BitSequence addition (__add__)
        bs1 = BitSequence([1, 0, 1])
        bs2 = BitSequence([0, 1])
        bs_sum = bs1 + bs2
        assert bs_sum.bits == [1, 0, 1, 0, 1], f"BitSequence add failed: {bs_sum.bits}"

    def test_encodedsequence_addition():
        # Test EncodedSequence addition (__add__)
        bv1 = BitVector(11)
        bv2 = BitVector(1, 3)
        es1 = EncodedSequence(bv1, uncompressed_length=7, alphabet_size=2)
        es2 = EncodedSequence(bv2, uncompressed_length=6, alphabet_size=2)
        es_sum = es1 + es2
        assert es_sum.data._value == 27 and es_sum.data._length == 7, f"EncodedSequence add failed: {es_sum.data._value}, {es_sum.data._length}"
        assert es_sum.uncompressed_length == 13, f"EncodedSequence add failed: {es_sum.uncompressed_length}"

    def test_all():
        test_bitvector_addition()
        test_bitsequence_addition()
        test_encodedsequence_addition()
        print("All addition tests passed.")

    test_all()