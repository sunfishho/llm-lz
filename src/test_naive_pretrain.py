from lz78_python import LZ78Encoder
from data_utils import convert_text_to_ascii_bits


# since LZ78Encoder expects a Sequence-compatible input, we need a fake Sequence of bits
class BitSequence:
    def __init__(self, bits):
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

def main():
    # Read in the text file
    with open("data/alice29.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Compress using LZ78
    ascii_bits = convert_text_to_ascii_bits(text)
    bit_sequence = BitSequence(ascii_bits)
    encoder = LZ78Encoder()
    encoded_bits, _, _ = encoder.encode_with_tree(bit_sequence)
    print(f"Original bit length (ASCII): {len(ascii_bits)}")
    print(f"Compressed (bit) size: {len(encoded_bits.data)}")

    N = 4096
    N0 = 2048
    k = 1024
    x_train = ascii_bits[:N]
    x_pretrain = x_train[:N0]
    x_eval = x_train[N0:]

    x_test = ascii_bits[N:]


    # Pretrain the LZ78 trie with the first k bits of x_train
    # pretrain_bits = x_train[:k]
    # pretrain_sequence = BitSequence(pretrain_bits)
    # lz_encoder = LZ78Encoder()
    # # Pretrain the encoder's trie
    # lz_encoder.pretrain(pretrain_sequence)

    # # Now, encode x_test using the pretrained trie
    # x_test_sequence = BitSequence(x_test)
    # encoded_test, _ = lz_encoder.encode_with_tree(x_test_sequence)

    # print(f"Length of x_test: {len(x_test)}")
    # print(f"Compressed size of x_test (with pretrained trie): {len(encoded_test.data)}")



if __name__ == "__main__":
    main()