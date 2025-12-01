import sys
import numpy as np
import argparse

from alice_compressor import AliceCompressorEnv, compute_reward, get_compression_length
from lz78 import LZ78Encoder, Sequence
from alice_cnn import AliceCNNPolicy, RewardFromObs, SeqCNNExtractor  # ensure pickle classes are available


MODEL_PATH = "model_saves_cnn/alice_compressor_policy_cnn_best.pkl"
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


def sample_rollout(model, env, rollout_length=100, deterministic=False):
    obs, _ = env.reset(seed=SEED)
    seq_chars = []
    reward = 0.0

    for _ in range(rollout_length):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        seq_chars.append(env.int_to_char[env._pretrain_sequence[-1]])
        if terminated or truncated:
            break

    return seq_chars, reward

def test_best_rollout(env):
    with open("model_saves_cnn/alice_compressor_policy_cnn_best_rollout.txt", "r", encoding="utf-8") as f:
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
    parser = argparse.ArgumentParser(description="Evaluate CNN policy on AliceCompressorEnv")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic rollout (default: stochastic policy rollout)",
    )
    args = parser.parse_args()
    deterministic = args.deterministic

    env = AliceCompressorEnv(seed=SEED)
    # rebuild deterministic critic to attach after load
    reward_evaluator = RewardFromObs(
        train_data=env.train_data,
        charmap=env.charmap,
        no_pretrain_len=env.no_pretrain_len,
        int_to_char=env.int_to_char,
        sample_k=10,
        seed=SEED,
    )
    try:
        # Make classes available under __main__ for unpickling
        sys.modules["__main__"].SeqCNNExtractor = SeqCNNExtractor
        sys.modules["__main__"].RewardFromObs = RewardFromObs
        sys.modules["__main__"].AliceCNNPolicy = AliceCNNPolicy
        model = AliceCNNPolicy.load(MODEL_PATH, device="mps")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    model.reward_evaluator = reward_evaluator

    seq_chars, rollout_reward = sample_rollout(model, env, rollout_length=env.max_pretrain_length, deterministic=deterministic)

    # Average rewards on train and test sets
    train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, seq_chars)
    test_no_pretrain_len = compute_no_pretrain_lengths(env.test_data, env.charmap)
    test_reward = evaluate_dataset_reward(env, env.test_data, test_no_pretrain_len, seq_chars)

    # Baseline: random 100-char substring from a random train chunk
    rand_chunk_idx = np.random.randint(len(env.train_data))
    rand_chunk = env.train_data[rand_chunk_idx]
    if len(rand_chunk) >= 100:
        start_idx = np.random.randint(0, len(rand_chunk) - 99)
        baseline_seq_chars = list(rand_chunk[start_idx : start_idx + 100])
    else:
        baseline_seq_chars = list(rand_chunk)
    baseline_train_reward = evaluate_dataset_reward(env, env.train_data, env.no_pretrain_len, baseline_seq_chars)
    baseline_test_reward = evaluate_dataset_reward(env, env.test_data, test_no_pretrain_len, baseline_seq_chars)

    print(f"Loaded CNN policy from {MODEL_PATH}")
    print(f"Rollout length: {len(seq_chars)}  Reward from rollout env: {rollout_reward:.3f}")
    print(f"Avg reward on train data: {train_reward:.3f}")
    print(f"Avg reward on test data: {test_reward:.3f}")
    print(f"Baseline (random 100-char slice from train chunk {rand_chunk_idx}) train reward: {baseline_train_reward:.3f}")
    print(f"Baseline (random 100-char slice from train chunk {rand_chunk_idx}) test reward: {baseline_test_reward:.3f}")
    print(f"Pretrain sequence:\n{seq_chars}")


if __name__ == "__main__":
    # main()
    env = AliceCompressorEnv(seed=SEED)
    print(test_best_rollout(env))
