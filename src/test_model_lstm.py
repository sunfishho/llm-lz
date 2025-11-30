import os
import sys
import re
import glob

import numpy as np

import agent
from agent import AliceLSTMPolicy
from alice_compressor import AliceCompressorEnv, compute_reward, get_compression_length
from lz78 import LZ78Encoder, Sequence\

num_layers = 3


def sample_rollout(model_path=None, rollout_length=100):
    env = AliceCompressorEnv(seed=78)
    model_path = f"model_policy/alice_compressor_policy_hidden_size=256_num_layers={num_layers}.pkl"
    print(f"Loading policy from {model_path}")
    # Load only the policy network saved during training
    # Models were saved when agent.py was run as __main__, so expose classes on __main__ for unpickling
    sys.modules["__main__"].SeqLenLSTMExtractor = agent.SeqLenLSTMExtractor
    sys.modules["__main__"].RewardFromObs = agent.RewardFromObs
    sys.modules["__main__"].AliceLSTMPolicy = agent.AliceLSTMPolicy
    policy = AliceLSTMPolicy.load(model_path, device="cpu")
    obs, _ = env.reset(seed=78)
    seq = []
    for _ in range(rollout_length):
        obs_input = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        action, _ = policy.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        seq.append(env._pretrain_sequence[-1])
        if terminated or truncated:
            break

    pretrain_seq = ''.join(seq)

    # Evaluate average reward on train data
    train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, seq)
    # Compute baselines and reward on held-out test data
    test_no_pretrain_len = compute_no_pretrain_lengths(env.test_data, env.charmap)
    test_reward = evaluate_dataset_reward(env, env.test_data, test_no_pretrain_len, seq)
    baseline_pretrain_chars = list(env.train_data[0][:100])
    baseline_test_reward = evaluate_dataset_reward(
        env, env.test_data, test_no_pretrain_len, baseline_pretrain_chars
    )

    print(f'chosen sequence: {seq}')
    print(f"Avg reward on train data: {train_reward}")
    print(f"Avg reward on test data: {test_reward}")
    print(f"Avg reward on test data (baseline first 100 chars of train[0]): {baseline_test_reward}")
    print(f"Pretrain string:\n{pretrain_seq}")


def compute_no_pretrain_lengths(data_splits, charmap):
    encoder = LZ78Encoder()
    return [get_compression_length(encoder.encode(Sequence(chunk, charmap=charmap))) for chunk in data_splits]


def evaluate_dataset_reward(env: AliceCompressorEnv, dataset, no_pretrain_len, seq_chars):
    rewards = []
    for i, chunk in enumerate(dataset):
        rewards.append(compute_reward(seq_chars, chunk, env.charmap, no_pretrain_len[i]))
    return sum(rewards) / len(rewards)


if __name__ == "__main__":
    sample_rollout()
