import gymnasium as gym
from alice_compressor import AliceCompressorEnv
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.utils import get_device, set_random_seed
from callbacks import RewardPlotCallback, SavePolicyCallback, RolloutPrintCallback, SaveBestRewardCallback
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

    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device = "mps",
        reward_evaluator: RewardFromObs | None = None,
        n_lstm_layers: int = 1,
    ):
        """
        Override policy loading so we can inject the LSTM architecture and reward evaluator.
        """
        device = get_device(device)
        saved_variables = torch.load(path, map_location=device, weights_only=False)
        constructor_kwargs = saved_variables["data"]
        constructor_kwargs["n_lstm_layers"] = n_lstm_layers
        if reward_evaluator is not None:
            constructor_kwargs["reward_evaluator"] = reward_evaluator
        model = cls(**constructor_kwargs)
        if reward_evaluator is not None:
            model.reward_evaluator = reward_evaluator
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model


model_dir = "model_saves_lstm"
plot_dir = "plots_lstm"
device = 'mps'
seed = 78
def train():
    set_random_seed(seed)
    env = AliceCompressorEnv(size_batch=5)

    save_callback = SavePolicyCallback(
        save_freq=2048,
        save_path=model_dir,
        prefix="alice_compressor_policy",
        suffix="lstm",
        verbose=1,
    )

    reward_plot_callback = RewardPlotCallback(
        save_path=os.path.join(plot_dir, "reward_plot.png"),
        smooth_window=1000,
        save_freq=8192,
        verbose=1,
    )

    rollout_print_callback = RolloutPrintCallback(env_fn=lambda: gym.make('alice-compressor-v0', seed=seed), print_freq=8192, rollout_length=300, verbose=1)

    reward_evaluator = RewardFromObs(
        train_data=env.train_data,
        charmap=env.charmap,
        no_pretrain_len=env.no_pretrain_len,
        int_to_char=env.int_to_char,
        sample_k=5,
        seed=seed,
    )

    save_best_callback = SaveBestRewardCallback(
        save_path=os.path.join(model_dir, "alice_compressor_policy_lstm_best"),
        verbose=1,
    )

    model = RecurrentPPO(
        policy=AliceManualCriticPolicy,
        env=env,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs={
            "reward_evaluator": reward_evaluator,
            "n_lstm_layers": 3,  # set LSTM layers to 3
        },
    )
    NUM_TIMESTEPS = 1_000_000
    iters = 0
    while iters < 5:
        model.learn(
            total_timesteps=NUM_TIMESTEPS, 
            reset_num_timesteps=False,
            callback=[save_callback, save_best_callback, rollout_print_callback, reward_plot_callback, ProgressBarCallback()],
        )

if __name__ == "__main__":
    train()
