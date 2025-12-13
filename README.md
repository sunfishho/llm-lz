# shared-dictionary-lz `src` directory guide

This directory contains experiments that use reinforcement learning to search for a shared dictionary that improves LZ78 compression of text (`data/alice29.txt`). Below is a map of the important files and what they do.

## Core environment
- `alice_compressor.py` defines the Gymnasium environment `alice-compressor-v0`. It loads the Alice text, builds a character map, exposes a `MultiDiscrete` observation (padded pretrain sequence + current length), and returns rewards equal to the compression gain (negative compressed bits) on sampled chunks using the LZ78 encoder. Helper utilities here: character set discovery, train/test chunking, reward computation, and a convenience `main` for smoke-testing the env.

## Training scripts (pretrain-sequence policies)
- `alice_cnn.py` trains a PPO agent with a 1D CNN feature extractor over the padded sequence. The critic is replaced with a deterministic reward evaluator (`RewardFromObs`) that mirrors `compute_reward` to remove critic learning noise. Saves policies under `model_saves_cnn/` and reward curves to `plots_cnn/`.
- `alice_mlp.py` trains a PPO agent with an MLP policy but the critic still comes from `RewardFromObs`. Outputs go to `model_saves_mlp/` and `plots_mlp/`.
- `alice_lstm.py` trains a recurrent PPO agent (`RecurrentPPO`) with a custom LSTM policy (`AliceManualCriticPolicy`) whose critic is the deterministic reward evaluator. Saves to `model_saves_lstm/` and `plots_lstm/`.
- Each training script uses callbacks from `callbacks.py` for periodic saving, reward plotting, greedy rollout logging, and “best reward so far” checkpoints.

## Evaluation scripts
- `test_cnn.py`, `test_mlp.py`, `test_lstm.py` load the corresponding saved policies, rebuild the deterministic critic, sample a rollout, and report average rewards on the train/test splits. They also compare against a random substring baseline and can read the best in-training rollout stored alongside the models.

## Support utilities
- `callbacks.py` implements:
  - `SavePolicyCallback` for periodic policy checkpoints,
  - `RewardPlotCallback` for smoothed reward curves,
  - `RolloutPrintCallback` for periodic greedy rollouts,
  - `SaveBestRewardCallback` to persist the policy and the rollout that achieved the best mean reward.
- `slow_lz_impl/` is a pure-Python LZ78 implementation (`lz78_python.py`, helpers in `data_classes.py` and `data_utils.py`) with tests such as `test_naive_pretrain.py` for algorithm validation and visualization. This is not used in the current implementation as it is orders of magnitude too slow.

## Data, artifacts, and variants
- `data/` holds `alice29.txt`
- `model_saves_*/` and `plots_*/` directories store trained policies, “best rollout” text files, and reward plots for each architecture.

Note: while most dependencies are mentioned in requirements.txt, to run the code in this library you must also follow the installation instructions in https://github.com/NSagan271/lz78_rust/tree/main in the pretrain-compression branch.