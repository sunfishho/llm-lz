import gymnasium as gym
import numpy as np
from lz78 import Sequence, LZ78Encoder, CharacterMap
from gymnasium import spaces
from gymnasium.envs.registration import register
from typing import List, Tuple, Optional
register(
    id='alice-compressor-v0',
    entry_point='alice_compressor:AliceCompressorEnv'
)

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
    encoder = LZ78Encoder()
    pretrain_data_str = ''.join(pretrain_data)
    encoder.pretrain(Sequence(pretrain_data_str, charmap=charmap))
    encoded = encoder.encode(Sequence(target_text, charmap=charmap))
    compression_len = get_compression_length(encoded)
    return -compression_len + no_pretrain_len


def generate_train_test(text: str, train_fraction: float = 0.7, chunk_size: int = 100, seed: int = 78) -> Tuple[List[str], List[str]]:
    """
    Generate train and test data from the Alice in Wonderland text file by chunking it 
    Args:
        text: Full text to split
        train_fraction: Fraction of data to use for training
        chunk_size: Size of each chunk in characters
        seed: Random seed for reproducibility
    Returns:
        Tuple of (train_data, test_data) where each is a list of text chunks

    """
    def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text) - chunk_size, chunk_size)]
    chunks = split_text_into_chunks(text, chunk_size)
    randomized_indices = np.random.RandomState(seed).permutation(len(chunks))
    train_data = [chunks[i] for i in randomized_indices[:int(train_fraction * len(chunks))]]
    test_data = [chunks[i] for i in randomized_indices[int(train_fraction * len(chunks)):]]
    return train_data, test_data

class AliceCompressorEnv(gym.Env):
    def __init__(self, seed: int = 78, max_pretrain_length: int = 1000, chunk_size: int = 100, size_batch: int = 5):
        """
        Args:
            seed: Random seed for reproducibility
            max_pretrain_length: Maximum length of pretraining sequence
            chunk_size: Size of each text chunk for training/testing
            size_batch: Number of chunks to sample for reward computation
        """
        super().__init__()
        with open("data/alice29.txt", "r", encoding="utf-8") as f:
            self.text = f.read()
        self.charmap = CharacterMap(self.text)
        self.init_char_operations()
        self.seed = seed
        self.chunk_size = chunk_size
        self.train_data, self.test_data = generate_train_test(self.text, train_fraction=0.7, chunk_size = self.chunk_size, seed=self.seed)
        self.max_pretrain_length: int = max_pretrain_length
        self.episode_idx: int = 0
        self._pretrain_sequence: List[int] = [] # string of characters in pretrained sequence
        self.size_batch = size_batch
        self.observation_space = spaces.MultiDiscrete([self.num_allowable_chars] * self.max_pretrain_length + [self.max_pretrain_length + 1])
        self.action_space = spaces.Discrete(self.num_allowable_chars)
        encoder = LZ78Encoder()

        # store the no pretrain length for each train data chunk
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

    def _get_obs(self):
        # returns the current pretrain sequence as a list of integers padded with 0s and the length of the sequence as the last element of the list
        padded_seq = [0] * self.max_pretrain_length + [len(self._pretrain_sequence)]
        padded_seq[:len(self._pretrain_sequence)] = self._pretrain_sequence.copy()
        return np.array(padded_seq)
        

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


    def step(self, action: int):
        """Execute one timestep within the environment.

        Args:
            action: An action

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._pretrain_sequence.append(action)
        pretrain_string = ''.join([self.int_to_char[val] for val in self._pretrain_sequence])
        terminated = (len(self._pretrain_sequence) == self.max_pretrain_length)
        # randomly choosing self.size_batch chunks of train data to compute the reward
        randomly_chosen_indices = self.np_random.integers(len(self.train_data), size=self.size_batch)
        rewards = [compute_reward(pretrain_string, self.train_data[i], self.charmap, self.no_pretrain_len[i]) for i in randomly_chosen_indices]
        # use the mean of the computed rewards as our final reward calculation
        reward = np.mean(rewards)
        return self._get_obs(), reward, terminated, False, {}

if __name__ == "__main__":
    env = gym.make('alice-compressor-v0')
    obs = env.reset()
    print(f"obs: {obs}")
    action = env.action_space.sample()
    print(f"action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    print(reward)
    print(terminated)
    print(truncated)
    print(info)