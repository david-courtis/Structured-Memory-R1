# Backbone Infrastructure

This directory contains Search-R1 examples and scripts. The core backbone Python packages (`verl/` and `search_r1/`) live at the repository root for Python import compatibility.

## Components

### `verl/` (root)
RL training framework built on Ray + PyTorch FSDP. Provides GRPO/PPO trainers, vLLM rollout, and distributed workers. Adapted from [veRL](https://github.com/volcengine/verl).

### `search_r1/` (root)
Retrieval infrastructure: sparse/dense/reranking servers, LLM agent generation utilities. Adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1).

### `backbone/scripts/`
Original Search-R1 training scripts (NQ/HotpotQA GRPO/PPO baselines). Not needed for StructMemoryR1 or Memory-R1 training.

### `backbone/example/`
Multi-node training configs and retriever launch examples from Search-R1.
