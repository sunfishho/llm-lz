import os
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from alice_compressor import AliceCompressorEnv, compute_reward
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from gymnasium import spaces

from callbacks import RewardPlotCallback, SavePolicyCallback, RolloutPrintCallback, SaveBestRewardCallback


class RewardFromObs:
    """
    Deterministic critic that mirrors alice_compressor.compute_reward using the padded
    sequence observation. Observations are MultiDiscrete vectors with the last element
    as the current length.
    """

    def __init__(self, train_data, charmap, no_pretrain_len, int_to_char, sample_k: int = 1, seed: int = 78):
        self.train_data = train_data
        self.charmap = charmap
        self.no_pretrain_len = no_pretrain_len
        self.int_to_char = int_to_char
        self.sample_k = sample_k
        self.rng = np.random.RandomState(seed)

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        obs_np = obs_tensor.detach().cpu().numpy()
        seq_part = obs_np[:, :-1]
        lengths = obs_np[:, -1].astype(int)
        rewards = []
        sample_indices = self.rng.choice(len(self.train_data), size=min(self.sample_k, len(self.train_data)), replace=False)
        for i, L in enumerate(lengths):
            L = int(L)
            pretrain_seq = [self.int_to_char[int(idx)] for idx in seq_part[i, :L]]
            total = 0.0
            for j in sample_indices:
                total += compute_reward(pretrain_seq, self.train_data[j], self.charmap, self.no_pretrain_len[j])
            rewards.append(total / len(sample_indices))
        return torch.tensor(rewards, device=obs_tensor.device, dtype=torch.float32).unsqueeze(1)


class SeqCNNExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for the padded sequence observation treated as a 1D signal.
    Architecture defined in __init__
    """

    def __init__(self, observation_space: spaces.MultiDiscrete, num_filters: int = 64, kernel_size: int = 3):
        # observation_space: [seq_len ... seq_len-1, length]
        self.seq_len = observation_space.nvec.shape[0] - 1
        features_dim = num_filters * 2
        super().__init__(observation_space, features_dim=features_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.len_embedding = nn.Embedding(observation_space.nvec[-1] + 1, num_filters)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq = observations[:, :-1].float().unsqueeze(1)  # (B,1,seq_len)
        lengths = observations[:, -1].long().clamp(min=0)
        seq_feat = self.conv(seq).squeeze(-1)
        len_feat = self.len_embedding(lengths)
        return torch.cat([seq_feat, len_feat], dim=1)


class AliceCNNPolicy(ActorCriticPolicy):
    """
    Feedforward policy that uses a CNN extractor and replaces the critic with deterministic reward.
    """

    def __init__(self, *args, reward_evaluator: RewardFromObs | None = None, **kwargs):
        reward_evaluator = kwargs.pop("reward_evaluator", reward_evaluator)
        kwargs.setdefault("features_extractor_class", SeqCNNExtractor)
        super().__init__(*args, **kwargs)
        self.reward_evaluator = reward_evaluator

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        actions, _, log_prob = super().forward(obs, deterministic=deterministic)
        values = self.reward_evaluator(obs)
        return actions, values.detach(), log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        _, log_prob, entropy = super().evaluate_actions(obs, actions)
        values = self.reward_evaluator(obs)
        return values.detach(), log_prob, entropy  

model_dir = "model_saves_cnn"
plot_dir = "plots_cnn"
device = "cpu"
seed = 78


def train():
    set_random_seed(seed)
    env = AliceCompressorEnv(size_batch=5)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    save_callback = SavePolicyCallback(
        save_freq=2048,
        save_path=model_dir,
        prefix="alice_compressor_policy",
        suffix="cnn",
        verbose=1,
    )

    reward_plot_callback = RewardPlotCallback(
        save_path=os.path.join(plot_dir, "reward_plot.png"),
        smooth_window=1000,
        save_freq=8192,
        verbose=1,
    )

    rollout_print_callback = RolloutPrintCallback(
        env_fn=lambda: gym.make("alice-compressor-v0", seed=seed),
        print_freq=8192,
        rollout_length=300,
        verbose=1,
    )

    save_best_callback = SaveBestRewardCallback(
        save_path=os.path.join(model_dir, "alice_compressor_policy_cnn_best.pkl"),
        verbose=1,
    )

    reward_evaluator = RewardFromObs(
        train_data=env.train_data,
        charmap=env.charmap,
        no_pretrain_len=env.no_pretrain_len,
        int_to_char=env.int_to_char,
        sample_k=5,
        seed=seed,
    )

    model = PPO(
        policy=AliceCNNPolicy,
        env=env,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs={
            "reward_evaluator": reward_evaluator,
        },
    )
    NUM_TIMESTEPS = 10_000_000
    iters = 0
    while iters < 5:
        model.learn(
            total_timesteps=NUM_TIMESTEPS,
            reset_num_timesteps=False,
            callback=[
                save_callback,
                save_best_callback,
                rollout_print_callback,
                reward_plot_callback,
                ProgressBarCallback(),
            ],
        )


if __name__ == "__main__":
    train()
