import sys
import numpy as np

from alice_compressor import AliceCompressorEnv, compute_reward, get_compression_length
from lz78 import LZ78Encoder, Sequence
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy


MODEL_PATH = "model_saves/alice_compressor_policy_hidden_size=256_num_layers=1.pkl"
SEED = 78


def compute_no_pretrain_lengths(chunks, charmap):
    encoder = LZ78Encoder()
    return [get_compression_length(encoder.encode(Sequence(chunk, charmap=charmap))) for chunk in chunks]


def evaluate_dataset_reward(env, dataset, no_pretrain_len, seq_chars):
    if seq_chars and isinstance(seq_chars[0], (int, np.integer)):
        seq_chars = [env.int_to_char[int(x)] for x in seq_chars]
    rewards = []
    for i, chunk in enumerate(dataset):
        rewards.append(compute_reward(seq_chars, chunk, env.charmap, no_pretrain_len[i]))
    return float(np.mean(rewards))


def sample_rollout(policy, env, rollout_length=100):
    obs, _ = env.reset(seed=SEED)
    lstm_states = None
    episode_starts = np.array([True], dtype=bool)
    seq_chars = []
    reward = 0.0

    for _ in range(rollout_length):
        action, lstm_states = policy.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=False,
        )
        obs, reward, terminated, truncated, _ = env.step(int(action))
        seq_chars.append(env.int_to_char[env._pretrain_sequence[-1]])
        episode_starts = np.array([terminated or truncated], dtype=bool)
        if terminated or truncated:
            break

    return seq_chars, reward


def main():
    env = AliceCompressorEnv(seed=SEED)
    try:
        policy = MlpLstmPolicy.load(MODEL_PATH, device="cpu")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    seq_chars, rollout_reward = sample_rollout(policy, env, rollout_length=env.max_pretrain_length)

    # Average rewards on train and test sets
    train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, seq_chars)
    test_no_pretrain_len = compute_no_pretrain_lengths(env.test_data, env.charmap)
    test_reward = evaluate_dataset_reward(env, env.test_data, test_no_pretrain_len, seq_chars)

    print(f"Loaded policy from {MODEL_PATH}")
    print(f"Rollout length: {len(seq_chars)}  Reward from rollout env: {rollout_reward:.3f}")
    print(f"Avg reward on train data: {train_reward:.3f}")
    print(f"Avg reward on test data: {test_reward:.3f}")
    print(f"Pretrain sequence:\n{seq_chars}")


if __name__ == "__main__":
    main()
