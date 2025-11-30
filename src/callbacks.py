
from stable_baselines3.common.callbacks import (
    BaseCallback,
)
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import os

class SavePolicyCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, prefix: str = "policy", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.prefix = prefix

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.prefix}_hidden_size=256_num_layers=5.pkl")
            self.model.policy.save(path)
            if self.verbose > 0:
                print(f"Saved policy to {path}")
        return True

class RewardPlotCallback(BaseCallback):
    """
    Collects per-step rewards and saves a smoothed reward plot at the end.
    """

    def __init__(self, save_path: str, smooth_window: int = 200, save_freq: int | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.smooth_window = smooth_window
        self.save_freq = save_freq
        self.rewards: list[float] = []
        self.timesteps: list[int] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.rewards.append(float(np.mean(rewards)))
            self.timesteps.append(self.num_timesteps)
        if self.save_freq and self.num_timesteps % self.save_freq == 0:
            self._save_plot()
        return True

    def _save_plot(self, suffix: str = "") -> None:
        if not self.rewards:
            return
        rewards_np = np.array(self.rewards)
        if len(rewards_np) >= self.smooth_window:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            smooth = np.convolve(rewards_np, kernel, mode="valid")
            timesteps_smoothed = self.timesteps[self.smooth_window - 1 :]
        else:
            smooth = rewards_np
            timesteps_smoothed = self.timesteps
        plt.figure(figsize=(8, 4))
        plt.plot(self.timesteps, self.rewards, alpha=0.3, label="reward per step")
        plt.plot(timesteps_smoothed, smooth, label=f"smoothed (window={self.smooth_window})")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        base, ext = os.path.splitext(self.save_path)
        out_path = f"{base}{suffix}{ext}"
        plt.savefig(out_path)
        plt.close()
        if self.verbose > 0:
            print(f"Saved reward plot to {out_path}")

    def _on_training_end(self) -> None:
        self._save_plot()

class RolloutPrintCallback(BaseCallback):
    """
    Every `print_freq` steps, runs a short greedy rollout and prints the pretrain string and reward.

    A dedicated evaluation environment is used so the training env is not mutated
    (avoids leaking `_pretrain_sequence` length into the collector).
    """

    def __init__(self, env_fn, print_freq: int = 16384, rollout_length: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.rollout_length = rollout_length
        self.env_fn = env_fn
        self.eval_env = env_fn()

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.print_freq == 0:
            obs, _ = self.eval_env.reset(seed=78)
            seq = []
            # Use unwrapped to access private attributes of the underlying environment
            base_env = self.eval_env.unwrapped
            reward = 0.0
            for _ in range(self.rollout_length):
                obs_input = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
                action, _ = self.model.policy.predict(obs_input, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action[0])
                seq.append(base_env._pretrain_sequence[-1])
                if terminated or truncated:
                    break
            print(f"\n[Rollout @ {self.num_timesteps} steps] len={len(seq)} reward={reward:.3f}\n{seq}\n")
            # Reset to avoid carrying over terminal state into the next callback invocation
            self.eval_env.reset(seed=78)
        return True
