from lz78_python import LZ78Encoder
from data_utils import convert_text_to_ascii_bits
from data_classes import BitVector, BitSequence, Sequence, EncodedSequence
import time


def main():
    # Read in the text file
    start_time = time.time()
    with open("data/alice29.txt", "r", encoding="utf-8") as f:
        text = f.read()
    read_file_time = time.time()
    # Compress using LZ78
    ascii_bits = convert_text_to_ascii_bits(text)
    bit_sequence = BitSequence(ascii_bits)
    encoder = LZ78Encoder()
    encoded_bits, _ = encoder.encode(bit_sequence)
    print(f"Original bit length (ASCII): {len(ascii_bits)}")
    print(f"Compressed (bit) size: {len(encoded_bits.data)}")
    compare_compressed_to_uncompressed_time = time.time()
    N = 4096
    N0 = 2048
    k = 1024
    x_train_lst = ascii_bits[:N]
    x_pretrain = BitSequence(x_train_lst[:N0])
    x_eval = BitSequence(x_train_lst[N0:])

    x_test = BitSequence(ascii_bits[N:])
    pretrained_encoder = LZ78Encoder()
    encoder = LZ78Encoder()
    pretrain_encoded, _ = pretrained_encoder.pretrain(x_pretrain, x_test)
    print(f'pretrained compressed length is {pretrain_encoded.compressed_len_bytes()}')
    encoded, _ = encoder.encode(x_test)
    print(f'not pretrained compressed length is {encoded.compressed_len_bytes()}')
    compare_pretrained_to_untrained_time = time.time()
    print(f'read_file_time: {read_file_time - start_time}, compare_compressed_to_uncompressed_time: {compare_compressed_to_uncompressed_time - read_file_time}, compare_pretrained_to_untrained_time: {compare_pretrained_to_untrained_time - compare_compressed_to_uncompressed_time}')

if __name__ == "__main__":
    main()