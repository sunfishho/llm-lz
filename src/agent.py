# used https://github.com/johnnycode8/airplane_boarding as a reference for the code below

from alice_compressor import AliceCompressorEnv
import gymnasium as gym
from alice_compressor import AliceCompressorEnv
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import  MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

import os

model_dir = "models"
log_dir = "logs"

def train():
    seed = 78
    model_dir = "model_saves"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('alice-compressor-v0')
    model = PPO("MultiInputPolicy", env=env, verbose=1, device = 'cpu', tensorboard_log=log_dir, seed = seed)

    NUM_TIMESTEPS = 1_000
    iters = 0
    while iters < NUM_TIMESTEPS:
        model.learn(total_timesteps=NUM_TIMESTEPS, reset_num_timesteps=False)
        model.save(os.path.join(model_dir, "alice_compressor.pkl"))

if __name__ == "__main__":
    train()

