import gymnasium as gym
import numpy as np
from tqdm import tqdm  # Progress bar
from lz78 import Sequence, LZ78Encoder, CharacterMap
from gymnasium import spaces
from gymnasium.envs.registration import register
from typing import List, Tuple, Optional
import pdb
from time import perf_counter
register(
    id='alice-compressor-v0',
    entry_point='alice_compressor:AliceCompressorEnv'
)
compute_reward_time = 0.0

def find_allowable_chars(text: str) -> List[str]:
    """
    Return all the characters that appear in text
    """
    return sorted(list(set(text)))

def get_compression_length(encoded) -> int:
    """
    Get compression length (bit length) from encoded sequence.
    
    Args:
        encoded: EncodedSequence object from LZ78 encoder
        
    Returns:
        Compression length in bits
    """
    return len(encoded.to_bytes())

def compute_reward(pretrain_data: str, target_text: str, charmap: CharacterMap, no_pretrain_len: int) -> int:
    """
    Compute compression length (reward) for given pretrain_data.
    
    Args:
        pretrain_data: String of pretraining characters
        target_text: Target text to compress (e.g., Alice in Wonderland)
        
    Returns:
        Negative compression length in bits (lower is better, so we'll negate for reward)
    """
    global compute_reward_time
    start = perf_counter()
    encoder = LZ78Encoder()
    pretrain_data_str = ''.join(pretrain_data)
    encoder.pretrain(Sequence(pretrain_data_str, charmap=charmap))
    encoded = encoder.encode(Sequence(target_text, charmap=charmap))
    compression_len = get_compression_length(encoded)
    compute_reward_time += perf_counter() - start
    # print(f"Pretrained length: {compression_len}, No pretrain length: {no_pretrain_len}")
    return -compression_len + no_pretrain_len


def generate_train_test(text: str, train_fraction: float = 0.7, chunk_size: int = 1000, seed: int = 78) -> Tuple[List[str], List[str]]:
    """
    Generate train and test data from the Alice in Wonderland text file by chunking it 
    """
    def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text) - chunk_size, chunk_size)]
    chunks = split_text_into_chunks(text, chunk_size)
    randomized_indices = np.random.RandomState(seed).permutation(len(chunks))
    train_data = [chunks[i] for i in randomized_indices[:int(train_fraction * len(chunks))]]
    test_data = [chunks[i] for i in randomized_indices[int(train_fraction * len(chunks)):]]
    return train_data, test_data

class AliceCompressorEnv(gym.Env):
    def __init__(self, seed: int = 78, max_pretrain_length: int = 100, chunk_size: int = 1000):
        super().__init__()
        with open("data/alice29.txt", "r", encoding="utf-8") as f:
            self.text = f.read()
        self.charmap = CharacterMap(self.text)
        self.init_char_operations()
        self.seed = seed
        self.chunk_size = chunk_size
        self.train_data, self.test_data = generate_train_test(self.text, train_fraction=0.7, chunk_size = self.chunk_size, seed=self.seed)
        self.max_pretrain_length = max_pretrain_length
        self.episode_idx = 0
        # surely the max length of the total compression is two times the length of the entire text?
        self._pretrain_sequence = [] # string of characters in pretrained sequence
        # although in theory this could break, i think in practice it will be fine
        # self.observation_space = spaces.Sequence(spaces.Discrete(self.num_allowable_chars))
        self.observation_space = spaces.Dict({
            'seq': spaces.Box(0, 1, shape=(self.max_pretrain_length, self.num_allowable_chars), dtype=np.int64),
            'len': spaces.Discrete(self.max_pretrain_length + 1),  # +1 to include 0
        })
        # we are first going to try just adding a character at a time
        self.action_space = spaces.Box(0, 1, shape=(self.num_allowable_chars,), dtype=np.int64)
        encoder = LZ78Encoder()
        self.no_pretrain_len = [get_compression_length(encoder.encode(Sequence(train_data, charmap=self.charmap))) for train_data in self.train_data]
    
    def init_char_operations(self):
        """
        Initialize character mappings
        """
        self.allowable_chars = find_allowable_chars(self.text)
        self.num_allowable_chars = len(self.allowable_chars)
        all_char_str = ''.join(self.allowable_chars)
        all_char_int = self.charmap.encode(all_char_str)
        self.char_to_int = {char: int for char, int in zip(self.allowable_chars, all_char_int)}
        self.int_to_char = {int: char for char, int in zip(self.allowable_chars, all_char_int)}
    
    def one_hot_encode_char(self, char: str) -> np.ndarray:
        """
        One-hot encode a character
        """
        idx = self.char_to_int[char]
        one_hot = np.zeros(self.num_allowable_chars)
        one_hot[idx] = 1
        return one_hot

    def _get_obs(self):
        # returns current pretrain sequencepadded_seq = np.zeros(self.max_pretrain_length, dtype=np.int64)
        padded_seq = np.zeros((self.max_pretrain_length, self.num_allowable_chars), dtype=np.int64)
        if len(self._pretrain_sequence) > 0:
            padded_seq[:len(self._pretrain_sequence)] = np.array([self.one_hot_encode_char(char) for char in self._pretrain_sequence], dtype=np.int64)
        return {
            'seq': padded_seq,
            'len': len(self._pretrain_sequence),
        }

    def reset(self, seed: Optional[int] = 78, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.episode_idx += 1
        self._pretrain_sequence = []
        return self._get_obs(), {}


    def step(self, action: int, size_batch: int = 5):
        """Execute one timestep within the environment.

        Args:
            action: A one-hot encoded action

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._pretrain_sequence.append(self.int_to_char[np.argmax(action)])
        terminated = (len(self._pretrain_sequence) == self.max_pretrain_length)

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        # convert the sequence of ints to a string that can be used to compute the reward
        # switch to just randomly choosing one chunk of train data to compute the reward
        randomly_chosen_index = np.random.randint(len(self.train_data), size=size_batch)
        rewards = [compute_reward(self._pretrain_sequence, self.train_data[i], self.charmap, self.no_pretrain_len[i]) for i in randomly_chosen_index]
        reward = np.mean(rewards)
        return self._get_obs(), reward, terminated, False, {}

if __name__ == "__main__":
    env = gym.make('alice-compressor-v0')
    obs = env.reset()
    print(obs)
    action = env.action_space.sample()
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    print(reward)
    print(terminated)
    print(truncated)
    print(info)
