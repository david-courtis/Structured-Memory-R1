# Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL

Implementation of the [Memory-R1 paper](https://arxiv.org/abs/2508.19828) (Yan et al., 2025) on top of the Search-R1 / veRL framework. Memory-R1 trains two RL agents to manage and use an external memory bank for multi-session dialogue QA:

- **Memory Manager**: Learns structured operations (ADD, UPDATE, DELETE, NOOP) over a memory bank
- **Answer Agent**: Learns to filter retrieved memories (Memory Distillation) and answer questions

Both agents are fine-tuned with GRPO using Exact Match reward on the [LoCoMo](https://github.com/snap-research/locomo) dataset.

---

## Quick Start

### Prerequisites

- 1x NVIDIA GPU with ≥24GB VRAM (RTX 3090/4090/A5000+)
- The `searchr1` conda environment from [SETUP.md](SETUP.md)
- (Optional) OpenAI API key for LLM-based fact extraction

```bash
conda activate searchr1
pip install scikit-learn  # Required for TF-IDF retrieval
```

> **Important:** All commands below use `conda run -n searchr1` to ensure the correct Python environment is used. This avoids issues where a system Python or another virtualenv shadows the conda env. The training scripts handle this automatically.

### 1. Build Training Data

```bash
# Uses heuristic fact extraction (no API key needed)
conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data

# OR: Use LLM extraction (matches paper, requires OPENAI_API_KEY)
export OPENAI_API_KEY='sk-...'
conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data --use_llm --llm_model gpt-5-nano
```

This downloads the LoCoMo dataset, parses it, and creates:

| File | Description | Samples |
|------|-------------|---------|
| `data/memory_r1/answer_agent/train.parquet` | Answer Agent train+val | 233 |
| `data/memory_r1/answer_agent/test.parquet` | Answer Agent test | 1307 |
| `data/memory_r1/memory_manager/train.parquet` | Memory Manager train+val | ~766 |
| `data/memory_r1/memory_manager/test.parquet` | Memory Manager test | ~4945 |

### 2. Train the Answer Agent (Stage 1)

```bash
bash train_answer_agent.sh
```

This trains the Answer Agent with GRPO on memory-augmented QA. The agent learns to select relevant memories and produce concise answers.

### 3. Train the Memory Manager (Stage 2)

```bash
# Without frozen Answer Agent (format-only reward)
bash train_memory_manager.sh

# With frozen Answer Agent (full EM reward, paper's approach)
# Pass the Stage 1 run directory; Stage 2 auto-resolves the latest actor/global_step_* checkpoint.
FROZEN_ANSWER_AGENT=verl_checkpoints/memory-r1-answer-agent-grpo-qwen2.5-0.5b \
  bash train_memory_manager.sh
```

The Memory Manager learns memory operations. When a frozen Answer Agent checkpoint is provided, it receives outcome-based reward: the EM score of the Answer Agent on the updated memory bank.

During training, validation periodically promotes the strongest checkpoint to `verl_checkpoints/<run_name>/best/`. The final post-training validation also updates that `best/` export if the last model is strongest.

---

## All Run Modes

### Data Construction

```bash
# Default: heuristic extraction, all defaults
conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data

# LLM extraction with custom model
conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data --use_llm --llm_model gpt-4o-mini

# Custom data/output directories
conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data --data_dir /path/to/locomo --output_dir /path/to/output
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_llm` | `False` | Use OpenAI API for fact extraction |
| `--llm_model` | `gpt-5-nano` | OpenAI model name |
| `--data_dir` | `data/locomo` | LoCoMo download directory |
| `--output_dir` | `data/memory_r1` | Output directory for parquets |

### Training

Both training scripts use `conda run -n searchr1` internally, so they work regardless of which Python your shell resolves to. They accept any veRL/Hydra override on the command line:

```bash
# Change base model
BASE_MODEL=Qwen/Qwen2.5-3B bash train_answer_agent.sh

# Adjust batch size and learning rate
bash train_answer_agent.sh data.train_batch_size=16 actor_rollout_ref.actor.optim.lr=5e-7

# Change number of GRPO candidates
bash train_answer_agent.sh actor_rollout_ref.rollout.n_agent=4

# Run validation only (no training)
bash train_answer_agent.sh +trainer.val_only=true

# Resume from checkpoint
bash train_answer_agent.sh actor_rollout_ref.model.path=verl_checkpoints/my-run/step_200
```

**Key training parameters:**

| Parameter | Answer Agent | Memory Manager | Paper |
|-----------|-------------|----------------|-------|
| `data.max_prompt_length` | 2048 | 2048 | 4096 |
| `data.max_response_length` | 256 | 512 | 2048 |
| `data.train_batch_size` | 4 | 4 | 128 |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | 1e-6 | 1e-6 |
| `actor_rollout_ref.rollout.temperature` | 1.0 | 1.0 | 1.0 |
| `trainer.total_training_steps` | 500 | 500 | ~200 |

> **Note:** Prompt/response lengths and batch sizes are reduced from the paper for 24GB VRAM. Increase them if you have more GPU memory.

### Memory Retrieval Server

For inference or multi-turn evaluation, you can run the memory retrieval server:

```bash
# Start with empty memory bank
conda run --no-capture-output -n searchr1 python3 -m memory_r1.retrieval.memory_server --port 8000

# Start with pre-loaded memories
conda run --no-capture-output -n searchr1 python3 -m memory_r1.retrieval.memory_server --memory_file memories.json --port 8000
```

The server exposes:
- `POST /retrieve` — Retrieve memories matching queries (Search-R1 compatible)
- `POST /update_memories` — Load/replace the memory bank
- `GET /status` — Check memory count

### Running Tests

```bash
# All tests
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/ -v

# Individual test files
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/test_memory_bank.py -v
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/test_prompts.py -v
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/test_data.py -v
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/test_reward.py -v
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/test_verl_integration.py -v
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Only with `--use_llm` | OpenAI API key for fact extraction |
| `CUDA_VISIBLE_DEVICES` | No (default: `0`) | GPU selection |
| `VLLM_ATTENTION_BACKEND` | No (default: `XFORMERS`) | vLLM attention backend |
| `FROZEN_ANSWER_AGENT` | No | Path to trained Answer Agent checkpoint for Stage 2 |
| `BASE_MODEL` | No (default: `Qwen/Qwen2.5-0.5B`) | Override in training scripts |

---

## End-to-End Pipeline

The complete Memory-R1 pipeline from raw data to trained models:

```
1. LoCoMo download & parse
   conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data
        |
        |-- data/memory_r1/answer_agent/{train,test}.parquet
        +-- data/memory_r1/memory_manager/{train,test}.parquet

2. Stage 1: Train Answer Agent
   bash train_answer_agent.sh
        |
        +-- verl_checkpoints/memory-r1-answer-agent-grpo-qwen2.5-0.5b/

3. Stage 2: Train Memory Manager (with frozen Answer Agent)
   FROZEN_ANSWER_AGENT=verl_checkpoints/.../memory-r1-answer-agent-... bash train_memory_manager.sh
        |
        +-- verl_checkpoints/memory-r1-manager-grpo-qwen2.5-0.5b/

4. Inference: Use both trained agents for memory-augmented QA
   (serve via memory_server.py + model inference)
```

---

## Troubleshooting

**`ModuleNotFoundError` (tensordict, pandas, sklearn, etc.)**

This almost always means the wrong Python is being used. Use `conda run` to guarantee the correct environment:
```bash
conda run --no-capture-output -n searchr1 python3 -m memory_r1.data.build_training_data
```

Common causes:
- A virtualenv (e.g., `arc-agi-3-agents`) is active and shadows the conda env's Python
- The system Python is being used instead of the conda env's
- Shell shows `🅒 searchr1` but `which python3` points elsewhere

The training scripts (`train_answer_agent.sh`, `train_memory_manager.sh`) handle this automatically via `conda run`.

**Out of GPU memory during training**
- Reduce `ppo_micro_batch_size` to 1
- Reduce `gpu_memory_utilization` to 0.2
- Reduce `data.max_prompt_length` to 1024
- Use a smaller model (`Qwen/Qwen2.5-0.5B`)

**`openai` package not found (when using `--use_llm`)**
```bash
conda run -n searchr1 pip install openai
```

**Parquet write errors (mixed types)**
- Ensure you've run the latest `build_training_data.py` — it coerces all QA answers to strings

**veRL import errors**
- Make sure you installed Search-R1 in development mode: `conda run -n searchr1 pip install -e .`
