import gymnasium as gym
from alice_compressor import AliceCompressorEnv
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.callbacks import ProgressBarCallback
from callbacks import RewardPlotCallback, SavePolicyCallback, RolloutPrintCallback
import os


model_dir = "model_saves"
plot_dir = "plots"
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

    model = RecurrentPPO("MlpLstmPolicy", env = env,verbose = 1, seed = seed,device = device)
    NUM_TIMESTEPS = 10_000_000
    iters = 0
    while iters < 5:
        model.learn(
            total_timesteps=NUM_TIMESTEPS, 
            reset_num_timesteps=False,
            callback=[save_callback, rollout_print_callback, reward_plot_callback, ProgressBarCallback()],
        )

if __name__ == "__main__":
    train()