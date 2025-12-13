import sys
import numpy as np

from alice_compressor import AliceCompressorEnv, compute_reward, get_compression_length
from lz78 import LZ78Encoder, Sequence
from alice_lstm import AliceManualCriticPolicy, RewardFromObs  # ensure pickle classes are available


MODEL_PATH = "model_saves_lstm/alice_compressor_policy_lstm.pkl"
SEED = 78


def make_reward_evaluator(env, sample_k: int = 5):
    """Rebuild the deterministic critic with the current env data."""
    return RewardFromObs(
        train_data=env.train_data,
        charmap=env.charmap,
        no_pretrain_len=env.no_pretrain_len,
        int_to_char=env.int_to_char,
        sample_k=sample_k,
        seed=SEED,
    )


def compute_no_pretrain_lengths(chunks, charmap):
    """
    Compute the compressed length of chunks of text without using a shared dictionary
    """
    encoder = LZ78Encoder()
    return [get_compression_length(encoder.encode(Sequence(chunk, charmap=charmap))) for chunk in chunks]


def evaluate_dataset_reward(env, dataset, no_pretrain_len, seq_chars):
    """
    Evaluate the reward of a sequence of characters on a dataset of chunks of text.
    """
    if seq_chars and isinstance(seq_chars[0], (int, np.integer)):
        seq_chars = [env.int_to_char[int(x)] for x in seq_chars]
    rewards = []
    for i, chunk in enumerate(dataset):
        rewards.append(compute_reward(seq_chars, chunk, env.charmap, no_pretrain_len[i]))
    return float(np.mean(rewards))


def sample_rollout(policy, env, rollout_length=100):
    """
    Sample a rollout from the policy.
    """
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

def test_best_rollout(env):
    """
    Test the best rollout from the model (as measured by the training set), read from the .txt file where it was saved.
    """
    with open("model_saves_lstm/alice_compressor_policy_lstm_best_rollout.txt", "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # Expect a Python list literal like ['a', ' ', 'b']; eval safely
    try:
        seq_chars = eval(raw)
    except Exception:
        # Fallback: treat as plain string
        seq_chars = list(raw)
    print(f"Pretrain sequence:\n{seq_chars}")
    train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, seq_chars)
    test_reward = evaluate_dataset_reward(env, env.test_data, env.no_pretrain_len, seq_chars)
    return train_reward, test_reward

def main():
    """
    Main function to evaluate the LSTM policy on the AliceCompressorEnv.
    """
    env = AliceCompressorEnv(seed=SEED)
    reward_evaluator = make_reward_evaluator(env)
    try:
        policy = AliceManualCriticPolicy.load(
            MODEL_PATH,
            device="mps",
            n_lstm_layers=3,
            reward_evaluator=reward_evaluator,
        )
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    # Ensure the loaded policy uses the freshly built reward evaluator.
    policy.reward_evaluator = reward_evaluator

    seq_chars, rollout_reward = sample_rollout(policy, env, rollout_length=env.max_pretrain_length)

    # Average rewards on train and test sets
    train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, seq_chars)
    test_no_pretrain_len = compute_no_pretrain_lengths(env.test_data, env.charmap)
    test_reward = evaluate_dataset_reward(env, env.test_data, test_no_pretrain_len, seq_chars)

    # print out information about the sample rollout and the rewards on the train and test sets
    print(f"Loaded policy from {MODEL_PATH}")
    print(f"Rollout length: {len(seq_chars)}  Reward from rollout env: {rollout_reward:.3f}")
    print(f"Avg reward on train data: {train_reward:.3f}")
    print(f"Avg reward on test data: {test_reward:.3f}")
    print(f"Pretrain sequence:\n{seq_chars}")


if __name__ == "__main__":
    # main()
    # Just test the best rollout 
    env = AliceCompressorEnv(seed=SEED)
    print(test_best_rollout(env))
    np.random.seed(SEED)
        # Baseline: random 100-char substring from a random train chunk
    rand_chunk_idx = np.random.randint(len(env.train_data))
    rand_chunk = env.train_data[rand_chunk_idx]
    if len(rand_chunk) >= 100:
        start_idx = np.random.randint(0, len(rand_chunk) - 99)
        baseline_seq_chars = list(rand_chunk[start_idx : start_idx + 100])
    else:
        baseline_seq_chars = list(rand_chunk)
    test_no_pretrain_len = compute_no_pretrain_lengths(env.test_data, env.charmap)
    baseline_train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, baseline_seq_chars)
    baseline_test_reward = evaluate_dataset_reward(env, env.test_data, test_no_pretrain_len, baseline_seq_chars)
    # Compute the reward associated with randomly selecting a 100-char substring from a random train chunk
    print(f"Baseline train reward: {baseline_train_reward:.3f}")
    print(f"Baseline test reward: {baseline_test_reward:.3f}")
