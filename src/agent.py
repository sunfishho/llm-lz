# used https://github.com/johnnycode8/airplane_boarding as a reference for the code below

from alice_compressor import AliceCompressorEnv, compute_reward
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
import torch
import torch.nn as nn
from gymnasium import spaces
from callbacks import SavePolicyCallback, RewardPlotCallback, RolloutPrintCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
import os

device = "cpu"
seed = 78
def linear_schedule(initial_value: float):
    """
    Linear decay of the learning rate from initial_value to 0 over training.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule

class RewardFromObs:
    """
    Deterministic reward evaluator that mirrors alice_compressor.compute_reward
    for the current partial sequence observation. To reduce cost, evaluates on
    a randomly chosen subset of k train chunks.
    """

    def __init__(self, train_data, charmap, no_pretrain_len, int_to_char, sample_k: int = 1, seed: int = 78):
        self.train_data = train_data
        self.charmap = charmap
        self.no_pretrain_len = no_pretrain_len
        self.int_to_char = int_to_char
        self.seed = seed
        self.sample_k = sample_k

    def __call__(self, seq_tensor: torch.Tensor, len_tensor: torch.Tensor) -> torch.Tensor:
        lengths = len_tensor.view(-1).long().cpu().numpy()
        seq_np = seq_tensor.cpu().numpy()
        rewards = []
        sample_indices = np.random.RandomState(self.seed).choice(
            len(self.train_data), size=min(self.sample_k, len(self.train_data)), replace=False
        )
        for i, L in enumerate(lengths):
            L = int(L)
            if L > 0:
                indices = seq_np[i, :L].argmax(axis=1)
                pretrain_seq = [self.int_to_char[int(idx)] for idx in indices]
            else:
                pretrain_seq = []
            total = 0.0
            for j in sample_indices:
                total += compute_reward(pretrain_seq, self.train_data[j], self.charmap, self.no_pretrain_len[j])
            rewards.append(total / len(sample_indices))
        return torch.tensor(rewards, device=seq_tensor.device, dtype=torch.float32).unsqueeze(1)

class SeqLenLSTMExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that runs the padded character sequence through an LSTM
    and concatenates it with an embedding of the sequence length.
    """

    def __init__(self, observation_space: spaces.Dict, lstm_hidden_size: int = 256, num_layers: int = 5):
        super().__init__(observation_space, features_dim=lstm_hidden_size)
        seq_space = observation_space["seq"]
        assert isinstance(seq_space, spaces.Box), "seq observation must be a Box space"
        self.num_chars = seq_space.shape[1]

        len_space = observation_space["len"]
        assert isinstance(len_space, spaces.Discrete), "len observation must be Discrete"

        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.num_chars,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout = 0.3,
        )
        self.len_embedding = nn.Embedding(len_space.n, self.lstm_hidden_size)
        self.projection = nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size)

    def forward(self, observations: dict) -> torch.Tensor:
        seq = observations["seq"].float()
        # SB3 may deliver `len` with an extra dim; flatten to 1D batch vector.
        lengths: torch.Tensor = observations["len"].long().view(-1).clamp(min=0)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)
        # Ensure batch dims match between seq and lengths
        if seq.size(0) != lengths.size(0):
            lengths = lengths[: seq.size(0)]

        # Sort by length for packing
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        seq_sorted = seq.index_select(0, sort_idx)

        packed = nn.utils.rnn.pack_padded_sequence(
            seq_sorted,
            lengths_sorted.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        lstm_out = hidden[-1]

        # Unsort back to the original batch order
        _, unsort_idx = sort_idx.sort()
        lstm_out = lstm_out.index_select(0, unsort_idx)
        lstm_out = lstm_out.masked_fill(lengths.eq(0).unsqueeze(1), 0)

        len_feat = self.len_embedding(lengths)
        combined = torch.cat([lstm_out, len_feat], dim=1)
        return torch.tanh(self.projection(combined))


class AliceLSTMPolicy(ActorCriticPolicy):
    """
    Multi-input PPO policy that uses an LSTM to encode the pretrain sequence.
    """

    def __init__(self, *args, reward_evaluator: RewardFromObs | None = None, **kwargs):
        # Avoid double-passing extractor kwargs when loading from checkpoints
        reward_evaluator = kwargs.pop("reward_evaluator", reward_evaluator)
        kwargs.setdefault("features_extractor_class", SeqLenLSTMExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"lstm_hidden_size": 256, "num_layers": 5})
        kwargs.setdefault("net_arch", [dict(pi=[64, 64], vf=[64, 64])])
        kwargs.setdefault("activation_fn", nn.ReLU)
        super().__init__(
            *args,
            **kwargs,
        )
        self.reward_evaluator = reward_evaluator

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=nn.ReLU
        )

    def _value_from_reward(self, obs: dict) -> torch.Tensor:
        if self.reward_evaluator is None:
            return self.value_net(self.extract_features(obs))
        return self.reward_evaluator(obs["seq"], obs["len"])

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._value_from_reward(obs)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._value_from_reward(obs)
        return values, log_prob, entropy


def train():
    seed = 78
    model_dir = "model_policy"
    log_dir = "logs"
    plots_dir = "plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('alice-compressor-v0') 
    reward_evaluator = RewardFromObs(
        train_data=env.train_data,
        charmap=env.charmap,
        no_pretrain_len=env.no_pretrain_len,
        int_to_char=env.int_to_char,
        sample_k=3,
        seed=seed,
    )
    model = PPO(
        AliceLSTMPolicy,
        env=env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        seed=seed,
        # learning_rate=linear_schedule(1e-3),
        policy_kwargs={"reward_evaluator": reward_evaluator},
    )

    NUM_TIMESTEPS = 10_000_000
    save_callback = SavePolicyCallback(
        save_freq=2048,
        save_path=model_dir,
        prefix="alice_compressor_policy",
        verbose=1,
    )
    reward_plot_callback = RewardPlotCallback(
        save_path=os.path.join(plots_dir, "reward_plot.png"),
        smooth_window=200,
        save_freq=8192,
        verbose=1,
    )
    rollout_print_callback = RolloutPrintCallback(env_fn=lambda: gym.make('alice-compressor-v0'), print_freq=2048, rollout_length=100, verbose=1)
    iters = 0
    while iters < 5:
        model.learn(
            total_timesteps=NUM_TIMESTEPS, 
            reset_num_timesteps=False,
            callback=[save_callback, reward_plot_callback, rollout_print_callback, ProgressBarCallback()],
        )
        iters += 1
    model.policy.save(os.path.join(model_dir, "alice_compressor_policy_hidden_size=256_num_layers=5.pkl"))
if __name__ == "__main__":
    train()
