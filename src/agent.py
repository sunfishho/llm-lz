# used https://github.com/johnnycode8/airplane_boarding as a reference for the code below

from alice_compressor import AliceCompressorEnv
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import  MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
import torch
import torch.nn as nn
from gymnasium import spaces

import os

model_dir = "models_lstm"
log_dir = "logs_lstm"


class SeqLenLSTMExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that runs the padded character sequence through an LSTM
    and concatenates it with an embedding of the sequence length.
    """

    def __init__(self, observation_space: spaces.Dict, lstm_hidden_size: int = 128):
        super().__init__(observation_space, features_dim=lstm_hidden_size)
        seq_space = observation_space["seq"]
        assert isinstance(seq_space, spaces.Box), "seq observation must be a Box space"
        self.num_chars = seq_space.shape[1]

        len_space = observation_space["len"]
        assert isinstance(len_space, spaces.Discrete), "len observation must be Discrete"

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=self.num_chars,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
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

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=SeqLenLSTMExtractor,
            features_extractor_kwargs={"lstm_hidden_size": 128},
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            activation_fn=nn.ReLU,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=nn.ReLU
        )


def train():
    seed = 78
    model_dir = "model_saves_lstm"
    log_dir = "logs_lstm"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('alice-compressor-v0')
    model = PPO(AliceLSTMPolicy, env=env, verbose=1, device='cpu', tensorboard_log=log_dir, seed=seed)

    NUM_TIMESTEPS = 10_000_000
    iters = 0
    while iters < NUM_TIMESTEPS:
        model.learn(total_timesteps=NUM_TIMESTEPS, reset_num_timesteps=False)
        if iters % 1000 == 0:
            model.save(os.path.join(model_dir, "alice_compressor.pkl"))
        iters += 1

if __name__ == "__main__":
    train()
