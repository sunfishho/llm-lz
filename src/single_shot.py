from lz78 import Sequence, LZ78Encoder, CharacterMap, BlockLZ78Encoder
from lz78 import encoded_sequence_from_bytes
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt

def compute_unpretrained_compression_length(target_text: str, charmap: CharacterMap) -> int:
    """
    Compute compression length of target_text without pretraining.
    """
    encoder = LZ78Encoder()
    encoded = encoder.encode(Sequence(target_text, charmap=charmap))
    return get_compression_length(encoded)

class PolicyNet(nn.Module):
    """Autoregressive policy network for generating pretrain_data."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 256, num_layers: int = 3):
        super(PolicyNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer to predict next character
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.output = nn.Softmax(dim=-1)
        
    def forward(self, char_indices: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass through the policy network.
        
        Args:
            char_indices: Tensor of shape (batch_size, seq_len) with character indices
            hidden: Optional LSTM hidden state tuple
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) with logits for each position
            hidden: Updated LSTM hidden state
        """
        # Embed characters
        embedded = self.embedding(char_indices)  # (batch_size, seq_len, embedding_dim)
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Get logits for next character prediction
        logits = self.output(self.linear(lstm_out))  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def get_log_probs(self, char_indices: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Get log probabilities for a sequence of characters.
        
        Args:
            char_indices: Tensor of shape (batch_size, seq_len) with character indices
            hidden: Optional LSTM hidden state tuple
            
        Returns:
            log_probs: Tensor of shape (batch_size, seq_len) with log probabilities
            hidden: Updated LSTM hidden state
        """
        logits, hidden = self.forward(char_indices, hidden)
        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # Extract log probs for the actual characters chosen
        # batch_size, seq_len = char_indices.shape
        log_probs_chosen = log_probs.gather(2, char_indices.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        
        return log_probs_chosen, hidden


def get_compression_length(encoded) -> int:
    """
    Get compression length (bit length) from encoded sequence.
    
    Args:
        encoded: EncodedSequence object from LZ78 encoder
        
    Returns:
        Compression length in bits
    """
    return len(encoded.to_bytes())


def compute_reward(pretrain_data: str, target_text: str, charmap: CharacterMap) -> float:
    """
    Compute compression length (reward) for given pretrain_data.
    
    Args:
        pretrain_data: String of pretraining characters
        target_text: Target text to compress (e.g., Alice in Wonderland)
        charmap: CharacterMap for encoding
        
    Returns:
        Compression length in bits (lower is better, so we'll negate for reward)
    """
    encoder = LZ78Encoder()
    encoder.pretrain(Sequence(pretrain_data, charmap=charmap))
    encoded = encoder.encode(Sequence(target_text, charmap=charmap))
    compression_length = get_compression_length(encoded)
    return compression_length


def sample_pretrain_data(
    policy_net: PolicyNet,
    char_to_idx: dict,
    idx_to_char: dict,
    max_length: int = 100,
    device: torch.device = None
) -> Tuple[str, torch.Tensor]:
    """
    Sample pretrain_data from the policy network autoregressively.
    Returns log_probs with gradients enabled for REINFORCE.
    
    Args:
        policy_net: Policy network
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        max_length: Maximum length of pretrain_data
        device: PyTorch device
        
    Returns:
        pretrain_data: Sampled string
        log_probs: Tensor of log probabilities for the sampled sequence (with gradients)
    """
    if device is None:
        device = next(policy_net.parameters()).device
    
    policy_net.train()  # Use train mode to enable gradients
    sampled_indices = []
    log_probs_list = []
    hidden = None
    
    # Start with a random character from vocabulary
    start_idx = np.random.randint(0, len(char_to_idx))
    sampled_indices.append(start_idx)
    current_char_idx = torch.tensor([[start_idx]], device=device, requires_grad=False)
    
    # Get log prob for first character (with gradients enabled)
    logits, hidden = policy_net(current_char_idx, hidden)
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    log_probs_list.append(log_probs[0, start_idx])
    
    # Sample remaining characters autoregressively
    for step in range(1, max_length):
        # Use the last sampled character as input
        logits, hidden = policy_net(current_char_idx, hidden)
        
        # Get log probabilities for next character (with gradients enabled)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, vocab_size)
        
        # Sample from the distribution (detach for sampling, but keep log_probs for gradient)
        probs = F.softmax(logits.detach()[:, -1, :], dim=-1)  # Detach for sampling
        sampled_idx = torch.multinomial(probs, 1)  # (1, 1)
        sampled_char_idx = sampled_idx.item()
        
        # Store log prob (with gradients) and character index
        log_probs_list.append(log_probs[0, sampled_char_idx])
        sampled_indices.append(sampled_char_idx)
        
        # Update input for next iteration (detached for next forward pass)
        # sampled_idx is already shape (1, 1) which is correct for LSTM input (batch_size, seq_len)
        current_char_idx = sampled_idx.detach()
    
    # Convert indices to characters
    sampled_chars = [idx_to_char[idx] for idx in sampled_indices]
    pretrain_data = ''.join(sampled_chars)
    log_probs_tensor = torch.stack(log_probs_list)
    
    return pretrain_data, log_probs_tensor


def build_vocabulary(text: str) -> Tuple[dict, dict, int]:
    """
    Build character vocabulary from text.
    
    Args:
        text: Input text
        
    Returns:
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        vocab_size: Size of vocabulary
    """
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(unique_chars)
    
    return char_to_idx, idx_to_char, vocab_size


class MovingAverageBaseline:
    """Moving average baseline for REINFORCE."""
    
    def __init__(self, alpha: float = 0.9):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher alpha = more weight on recent values.
        """
        self.alpha = alpha
        self.baseline = None
        self.recent_values = deque(maxlen=100)  # Keep last 100 values
    
    def update(self, value: float):
        """Update baseline with new value."""
        self.recent_values.append(value)
        if self.baseline is None:
            self.baseline = value
        else:
            self.baseline = self.alpha * value + (1 - self.alpha) * self.baseline
    
    def get(self) -> float:
        """Get current baseline value."""
        if self.baseline is None:
            return 0.0
        return self.baseline

def save_model(policy_net: PolicyNet, path: str):
    """Save policy network to file."""
    torch.save(policy_net.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path: str) -> PolicyNet:
    """Load policy network from file."""
    policy_net = PolicyNet(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2)
    policy_net.load_state_dict(torch.load(path))
    return policy_net   

def train_reinforce(
    policy_net: PolicyNet,
    target_text: str,
    charmap: CharacterMap,
    char_to_idx: dict,
    idx_to_char: dict,
    num_episodes: int = 1000,
    max_pretrain_length: int = 100,
    learning_rate: float = 0.01,
    device: torch.device = None,
    print_every: int = 50
) -> Tuple[str, List[float]]:
    """
    Train policy network using REINFORCE with baseline.
    
    Args:
        policy_net: Policy network to train
        target_text: Target text to compress
        charmap: CharacterMap for encoding
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        num_episodes: Number of training episodes
        max_pretrain_length: Maximum length of pretrain_data
        learning_rate: Learning rate for optimizer
        device: PyTorch device
        print_every: Print progress every N episodes
        
    Returns:
        best_pretrain_data: Best pretrain_data found
        rewards_history: List of rewards over training
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy_net = policy_net.to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    baseline = MovingAverageBaseline(alpha=0.9)
    
    best_pretrain_data = None
    best_reward = float('inf')  # Lower compression length is better
    rewards_history = []
    unpretrained_length = compute_unpretrained_compression_length(target_text, charmap)

    naive_pretrained_length = compute_reward(target_text[:max_pretrain_length], target_text, charmap)
    
    for episode in range(num_episodes):
        # Sample pretrain_data from policy
        pretrain_data, log_probs = sample_pretrain_data(
            policy_net, char_to_idx, idx_to_char, max_pretrain_length, device
        )
        
        # Compute reward (compression length)
        reward = compute_reward(pretrain_data, target_text, charmap)
        rewards_history.append(reward)
        
        # Update baseline
        baseline.update(reward)
        baseline_value = baseline.get()
        
        # Compute advantage
        advantage = reward - baseline_value
        
        # REINFORCE loss: -sum(log_probs) * advantage
        # We negate because we want to maximize reward (minimize compression length)
        loss = -torch.sum(log_probs) * advantage
        
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track best
        if reward < best_reward:
            best_reward = reward
            best_pretrain_data = pretrain_data
        
        # Print progress
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Current pretrain_data: {pretrain_data[:50]}...")
            print(f"  Compression length: {reward:.2f}")
            print(f"  Baseline: {baseline_value:.2f}")
            print(f"  Advantage: {advantage:.2f}")
            print(f"  Best compression length: {best_reward:.2f}")
            print(f"  Best pretrain_data: {best_pretrain_data[:50]}...")
            print(f"  No pretrain compression length: {unpretrained_length:.2f}")
            print(f"  Naive pretrained compression length: {naive_pretrained_length:.2f}")
            print()

        if (episode + 1) % 100 == 0:
            save_model(policy_net, f"model_saves/single_shot_model.pth")
    
    return best_pretrain_data, rewards_history


if __name__ == "__main__":
    # Load target text
    with open("data/alice29.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Build character map and vocabulary
    charmap = CharacterMap(text)
    char_to_idx, idx_to_char, vocab_size = build_vocabulary(text)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(text)}")
    
    # Create policy network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = PolicyNet(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2)
    
    # Train using REINFORCE
    best_pretrain_data, rewards_history = train_reinforce(
        policy_net=policy_net,
        target_text=text,
        charmap=charmap,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        num_episodes=2000,
        max_pretrain_length=100,
        learning_rate=0.01,
        device=device,
        print_every=50
    )

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, label="Compression Length (Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Compression Length (bits)")
    plt.title("REINFORCE Training: Compression Length over Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/single_shot_reinforce_training.png")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best pretrain_data: {best_pretrain_data}")
    print(f"Best compression length: {min(rewards_history):.2f}")
    print("="*50)
