# StructMemoryR1

Long-horizon conversational context access and updates over tree-structured
memory with reinforcement learning. This repository accompanies the paper
[*Struct Memory-R1*](Struct_Memory_R1.pdf), which trains three cooperating
GRPO agents to **read** from and **write** to a tree-structured memory bank
built from raw multi-session dialogue.

The three agents:

- **Memory Manager** (π_θ_M) — emits CRUD operations
  (`ADD`, `UPDATE`, `DELETE`, `NONE`) that update the memory tree as new
  dialogue turns arrive.
- **Retrieve Agent** (π_θ_R) — emits a JSON search plan over the tree
  schema, selecting branches and leaves rather than scoring every memory
  independently.
- **Answer Agent** (π_θ_A) — produces the final short answer from the
  retrieved leaves.

Training is staged: the Answer Agent is trained first and then frozen; the
Retrieve Agent is trained next with the frozen Answer Agent as the reward
oracle; the Memory Manager is trained last with both downstream agents
frozen. All three are optimised with Group Relative Policy Optimization
(GRPO) on a Qwen2.5-3B-Instruct backbone.

## Headline results on LoCoMo

1:1:8 train / val / test split (152 / 81 / 1,307 QA pairs), adversarial
category excluded, evaluated with token F1 (**F1**), BLEU-1 (**B1**), and
GPT-4o-mini judge accuracy (**J**). Best non-oracle in **bold**; oracle in
*italics*.

|                         | Overall F1 | Overall J | Multi-hop F1 | Temporal F1 | Single-hop F1 | Open-domain F1 |
| ----------------------- | ---------- | --------- | ------------ | ----------- | ------------- | -------------- |
| Full Context            | 24.2       | 45.7      | 19.7         | 16.2        | 23.9          | 27.0           |
| *Oracle*                | *31.3*     | *60.8*    | *34.3*       | *35.4*      | *31.6*        | *29.5*         |
| RAG (BM25)              | 29.9       | 49.8      | 19.0         | 11.2        | 22.6          | **38.7**       |
| RAG (Embedding)         | 29.5       | 49.6      | 21.1         | 12.1        | 21.8          | 37.2           |
| Semantic XPath          | 18.5       | 39.8      | 25.9         | 29.8        | 21.3          | 13.4           |
| Memory-R1 (re-impl.)    | 27.4       | 53.9      | 22.5         | 20.8        | **27.7**      | 30.0           |
| **StructMemoryR1**      | **30.4**   | **54.4**  | **31.5**     | **34.6**    | 27.1          | 30.6           |

Tree-structured retrieval delivers **+9.0 F1 on multi-hop** and **+13.8 F1
on temporal** over the flat Memory-R1 baseline; full ablations are in the
paper.

## Repository layout

```text
.
├── struct_memory_r1/           # Main contribution: 3-agent RL system
│   ├── agents/                 #   memory_manager.py, retrieve_agent.py
│   ├── data/                   #   LoCoMo + Struct-LoCoMo parquet builder
│   ├── retrieval/              #   FastAPI memory retrieval server
│   ├── reward/                 #   EM / token-F1 / answer extraction
│   ├── scripts/                #   per-stage and full-pipeline shell scripts
│   ├── tests/                  #   pytest suite
│   ├── memory_bank.py          #   Tree T=(V,E), CRUD + retrieval executor
│   ├── prompts.py              #   System prompts and training prompt builders
│   └── llm_extract.py          #   Optional LLM fact extractor (OpenAI API)
├── memory_r1/                  # Faithful flat-memory baseline (2-agent)
│   └── ...                     #   same layout as struct_memory_r1/
├── structured_locomo_trees/    # Struct-LoCoMo: 10 conversation XML trees
│   │                           #   plus locomo10.json (LoCoMo source)
│   └── conv-*.xml
├── verl/                       # GRPO/PPO trainer (adapted from veRL)
├── search_r1/                  # Rollout / generation utilities used by verl/
├── legacy/                     # Pre-merger forks, not used at runtime
├── pyproject.toml              # Canonical package metadata
├── requirements.txt            # Pinned runtime deps for the training box
└── Struct_Memory_R1.pdf        # The paper
```

`verl/` and `search_r1/` are adapted from
[veRL](https://github.com/volcengine/verl) and
[Search-R1](https://github.com/PeterGriffinJin/Search-R1). Anything under
`legacy/` is kept only for historical reference — see
[legacy/README.md](legacy/README.md).

## Installation

```bash
conda create -n structmemory python=3.10 -y
conda activate structmemory

# Local install
pip install -e .

# CUDA-only / hardware-specific deps not in pyproject.toml
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

`flash-attn` and `vllm` require a CUDA-enabled GPU; the rest installs on
CPU. Training was developed against PyTorch 2.4, vLLM 0.6.3, and Ray 2.10
on a single NVIDIA H200.

## Data

LoCoMo's source JSON and the Struct-LoCoMo tree files ship in
[structured_locomo_trees/](structured_locomo_trees/). The loader checks
that directory first, then falls back to downloading
`locomo10.json` from
[snap-research/locomo](https://github.com/snap-research/locomo).

Build the per-stage training parquets:

```bash
# All three stages at once
python -m struct_memory_r1.data.build_training_data --stage all

# Or one stage at a time
python -m struct_memory_r1.data.build_training_data --stage answer_agent
python -m struct_memory_r1.data.build_training_data --stage retrieve_agent
python -m struct_memory_r1.data.build_training_data --stage memory_manager
```

Outputs land in `data/struct_memory_r1/<stage>/{train,test}.parquet`. Add
`--use_llm` to swap the heuristic fact extractor for the paper's LLM
extractor (requires `OPENAI_API_KEY`).

The Memory-R1 baseline has its own builder:

```bash
python -m memory_r1.data.build_training_data
```

## Training

Each stage is a `verl.trainer.main_ppo` invocation wrapped in a shell
script. The scripts auto-build the parquet if missing.

```bash
# Stage 1 — Answer Agent
bash struct_memory_r1/scripts/train_answer_agent.sh

# Stage 2 — Retrieve Agent (point at the Stage 1 checkpoint)
FROZEN_ANSWER_AGENT=verl_checkpoints/<stage1-run> \
bash struct_memory_r1/scripts/train_retrieve_agent.sh

# Stage 3 — Memory Manager (point at Stage 1 + Stage 2 checkpoints)
FROZEN_ANSWER_AGENT=verl_checkpoints/<stage1-run> \
FROZEN_RETRIEVER=verl_checkpoints/<stage2-run> \
bash struct_memory_r1/scripts/train_memory_manager.sh
```

A combined cloud-GPU pipeline that runs all three stages back-to-back:

```bash
bash struct_memory_r1/scripts/nebius_train.sh --stage all
```

Memory-R1 (baseline) trains in two stages:

```bash
bash memory_r1/scripts/nebius_train.sh --stage both
```

Both pipelines write per-step checkpoints under `verl_checkpoints/` and log
to `logs/`. End-to-end training of all three StructMemoryR1 stages takes
~150 GPU-hours on a single H200.

### Hyperparameters that matter

| Knob                       | Default               | Where                           |
| -------------------------- | --------------------- | ------------------------------- |
| `BASE_MODEL`               | `Qwen/Qwen2.5-3B-Instruct` | env var on every `train_*.sh` |
| `data.train_batch_size`    | 32                    | `train_*.sh`                    |
| `actor_rollout_ref.actor.optim.lr` | 1e-5 (Stage 1) / 1e-6 (Stage 2–3) | `train_*.sh` |
| `actor_rollout_ref.rollout.n_agent` | 8 (GRPO group size) | `train_*.sh` |
| `trainer.total_training_steps`     | 500 (750 for Stage 1) | `train_*.sh` |

## Retrieval server

The Retrieve Agent and Memory Manager training loops call into a FastAPI
process that owns the memory tree and executes search plans. Start it
before the corresponding training stage:

```bash
python -m struct_memory_r1.retrieval.memory_server  # listens on :8000
```

The Memory-R1 baseline ships an analogous TF-IDF server at
`memory_r1.retrieval.memory_server`.

## Tests

```bash
python -m pytest struct_memory_r1/tests/ -v
python -m pytest memory_r1/tests/ -v
```

The unit suites (memory bank CRUD, prompt builders, reward functions,
training-data construction) run without GPUs. The `test_verl_integration`
tests load veRL and therefore require the full training environment
(`tensordict`, `vllm`, `ray`, …).

## Reproducing the paper

1. `pip install -e . && pip install -r requirements.txt`
2. `python -m struct_memory_r1.data.build_training_data --stage all`
3. `bash struct_memory_r1/scripts/nebius_train.sh --stage all`
4. `bash memory_r1/scripts/nebius_train.sh --stage both` for the baseline
5. Evaluate predicted answers with the LLM-as-judge prompt in
   [struct_memory_r1/prompts.py](struct_memory_r1/prompts.py) (`JUDGE_PROMPT_TEMPLATE`)
   against GPT-4o-mini, as in the paper.

Trained checkpoints are not committed here due to size; the camera-ready
will publish a Google Drive link.

## Limitations (mirroring the paper, §8)

- Tree structure currently comes from oracle human-verified Struct-LoCoMo
  XMLs; full end-to-end Memory Manager RL is in progress.
- The Memory Manager performs CRUD updates only; structural restructuring
  ops (`SPLIT_TOPIC`, `MERGE_TOPIC`, …) are scaffolded in
  `memory_bank.py` but not trained.
- The Retrieve Agent emits a single search plan per question rather than
  acting interactively level-by-level.
- No hybrid dense + structural retrieval.
- Evaluation is limited to LoCoMo.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{structmemoryr1,
  title  = {Struct Memory-R1: Long-Horizon Conversational Context Access and Updates over Tree-Structured Memory with Reinforcement Learning},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements

`verl/` and `search_r1/` are adapted from the
[veRL](https://github.com/volcengine/verl) and
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) codebases. The
Answer Agent prompts and our Memory-R1 baseline re-implementation follow
[Memory-R1](https://arxiv.org/abs/2508.19828). LoCoMo is from
[snap-research/locomo](https://github.com/snap-research/locomo).
