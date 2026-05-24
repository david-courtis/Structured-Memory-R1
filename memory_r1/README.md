# Memory-R1 (Baseline)

Re-implementation of **Memory-R1**, a flat-memory RL system for long-term dialogue QA. Memory-R1 does not provide a public codebase; this is our from-scratch re-implementation based on the original paper, retrained on the same base model (Qwen2.5-3B-Instruct) and data split for fair comparison with StructMemoryR1.

Uses the same two-stage GRPO training pipeline as StructMemoryR1 but with a flat (unstructured) memory bank instead of a tree.

For detailed architecture and design notes, see [docs/MEMORY_R1_ARCHITECTURE.md](docs/MEMORY_R1_ARCHITECTURE.md) and [docs/MEMORY_R1_README.md](docs/MEMORY_R1_README.md).

## Module Structure

```
memory_r1/
├── memory_bank.py          # Flat memory bank (list of text facts)
├── prompts.py              # Prompt templates
├── llm_extract.py          # Fact extraction from dialogue
├── agents/
│   └── memory_manager.py   # Memory Manager agent + reward function
├── data/
│   ├── build_training_data.py   # Build LoCoMo training parquets
│   └── locomo_loader.py         # LoCoMo dataset loader
├── retrieval/
│   └── memory_server.py    # TF-IDF memory retrieval server
├── reward/
│   └── em_reward.py        # EM / F1 reward scoring
├── scripts/                # Training scripts
│   ├── train_answer_agent.sh    # Stage 1 (single GPU)
│   ├── train_memory_manager.sh  # Stage 2 (single GPU)
│   └── nebius_train.sh          # Full pipeline (cloud GPU)
├── docs/                   # Architecture and design notes
└── tests/
```

## Data Preparation

```bash
python -m memory_r1.data.build_training_data
```

Outputs:
- `data/memory_r1/answer_agent/train.parquet`
- `data/memory_r1/answer_agent/test.parquet`
- `data/memory_r1/memory_manager/train.parquet`
- `data/memory_r1/memory_manager/test.parquet`

## Training

**Stage 1 — Answer Agent (single GPU):**
```bash
bash memory_r1/scripts/train_answer_agent.sh
```

**Stage 2 — Memory Manager (single GPU):**
```bash
bash memory_r1/scripts/train_memory_manager.sh
```

**Cloud (H200, full pipeline):**
```bash
bash memory_r1/scripts/nebius_train.sh --stage both --model Qwen/Qwen2.5-3B-Instruct
```

Key parameters (set via env vars or `--flag`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID |
| `TRAIN_BATCH_SIZE` | `4` | Global training batch size |
| `GPU_MEM_UTIL` | `0.4` | vLLM GPU memory fraction |
| `MAX_PROMPT_LENGTH` | `4096` | Max tokens for Stage 1 prompts |
| `MAX_PROMPT_LENGTH_STAGE2` | `8192` | Max tokens for Stage 2 prompts |
| `TOTAL_STEPS` | `500` | Training steps per stage |

## Retrieval Server

Start before training (required for rollout):
```bash
python -m memory_r1.retrieval.memory_server
```

Runs on `http://localhost:8000`. Endpoint: `POST /retrieve`.

## Tests

```bash
python -m pytest memory_r1/tests/ -v
```
