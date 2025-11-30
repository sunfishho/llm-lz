import gymnasium as gym
from alice_compressor import AliceCompressorEnv
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import ProgressBarCallback
from callbacks import RewardPlotCallback, SavePolicyCallback, RolloutPrintCallback
from alice_compressor import compute_reward
import numpy as np
import torch
import os


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


class AliceManualCriticPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent policy that keeps the action network but replaces the critic
    with a deterministic reward evaluator from alice_compressor.
    """

    def __init__(self, *args, reward_evaluator: RewardFromObs | None = None, **kwargs):
        reward_evaluator = kwargs.pop("reward_evaluator", reward_evaluator)
        super().__init__(*args, **kwargs)
        self.reward_evaluator = reward_evaluator

    def forward(self, obs: torch.Tensor, lstm_states, episode_starts, deterministic: bool = False):
        actions, _, log_prob, new_lstm_states = super().forward(obs, lstm_states, episode_starts, deterministic)
        values = self.reward_evaluator(obs)
        return actions, values.detach(), log_prob, new_lstm_states

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, lstm_states, episode_starts):
        _, log_prob, entropy = super().evaluate_actions(obs, actions, lstm_states, episode_starts)
        values = self.reward_evaluator(obs)
        return values.detach(), log_prob, entropy


model_dir = "model_saves_2"
plot_dir = "plots_2"
device = 'cpu'
seed = 78
def train():
    env = AliceCompressorEnv()

    save_callback = SavePolicyCallback(
        save_freq=512,
        save_path=model_dir,
        prefix="alice_compressor_policy",
        verbose=1,
    )

    reward_plot_callback = RewardPlotCallback(
        save_path=os.path.join(plot_dir, "reward_plot.png"),
        smooth_window=200,
        save_freq=8192,
        verbose=1,
    )

    rollout_print_callback = RolloutPrintCallback(env_fn=lambda: gym.make('alice-compressor-v0', seed=seed), print_freq=1024, rollout_length=100, verbose=1)

    reward_evaluator = RewardFromObs(
        train_data=env.train_data,
        charmap=env.charmap,
        no_pretrain_len=env.no_pretrain_len,
        int_to_char=env.int_to_char,
        sample_k=3,
        seed=seed,
    )

    model = RecurrentPPO(
        AliceManualCriticPolicy,
        env=env,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs={"reward_evaluator": reward_evaluator},
    )
    NUM_TIMESTEPS = 1_000_000
    iters = 0
    while iters < 5:
        model.learn(
            total_timesteps=NUM_TIMESTEPS, 
            reset_num_timesteps=False,
            callback=[save_callback, rollout_print_callback, reward_plot_callback, ProgressBarCallback()],
        )

if __name__ == "__main__":
    train()
