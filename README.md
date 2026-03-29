# StructMemoryR1: Long-Horizon Conversational Context Access and Updates over Tree-Structured Memory with Reinforcement Learning

This repository contains the code and data for **StructMemoryR1**, a three-stage reinforcement learning framework for training LLM agents to maintain and query structured memory across long-term multi-session conversations.

## Overview

StructMemoryR1 trains three cooperating RL agents on the [LoCoMo](https://snap.stanford.edu/locomo/) dataset:

- **Memory Manager** (π_θM) — induces and updates a tree-structured memory bank via CRUD operations (ADD, UPDATE, DELETE, NONE)
- **Retrieve Agent** (π_θR) — plans a structured search over the memory tree schema to select relevant evidence
- **Answer Agent** (π_θA) — answers questions from the retrieved evidence

All three components are fine-tuned with GRPO using downstream QA reward. Training is staged: Answer Agent first (then frozen), then Retrieve Agent, then Memory Manager — ensuring stable reward signals at each phase.

```
Conversation Sessions
        │
        ▼
┌───────────────────┐       ┌──────────────────┐
│  Memory Manager   │──────▶│   Memory Bank    │
│  (π_θM)           │  ADD/ │  (rooted tree)   │
└───────────────────┘ UPD/  └────────┬─────────┘
        ▲             DEL            │ schema
        │                            ▼
        │             ┌──────────────────────┐
        │             │   Retrieve Agent     │──── search plan ──▶ evidence
        │             │   (π_θR)             │
        │             └──────────┬───────────┘
        │                        │ retrieved memories
        │                        ▼
        │             ┌──────────────────────┐
        └─────reward──│   Answer Agent       │
          F1(answer)  │   (π_θA)             │
                      └──────────────────────┘
                               │
                               ▼
                          Final Answer
```

## Dataset

We evaluate on **LoCoMo**, a long-horizon conversational memory benchmark containing 10 multi-session dialogues with 1,540 non-adversarial QA pairs across four question categories: single-hop, multi-hop, temporal, and open-domain. Following the Memory-R1 evaluation protocol, we exclude the adversarial category. We use a 1:1:8 train/validation/test split, yielding **152 training**, **81 validation**, and **1,307 test** QA pairs.

### Struct-LoCoMo

Since LoCoMo provides raw multi-session dialogues without any structured memory representation, we manually convert each sample into a tree-structured memory following the schema used by our framework. Each dialogue is organized into a four-level hierarchy:

```
Root → Conversation → Speaker → Session → Observation
```

Each leaf node corresponds to an atomic factual observation extracted from a single dialogue turn. For example, a fact uttered by John in Session 13 is placed at:

```
Root > Conversation_v1 > John > Session_13 > Observation_1
```

This conversion is performed by a prompted LLM and verified by human annotators to ensure structural correctness and factual fidelity. The resulting structured memory trees are located in [`structured_locomo_trees/`](structured_locomo_trees/).

## Model and Training

- **Base model**: Qwen2.5-3B-Instruct for all trainable components
- **Hardware**: Single NVIDIA H200 GPU (141 GB HBM3e), 16 CPU cores
- **Optimizer**: GRPO (Group Relative Policy Optimization)
- **Training steps**: 500 per agent
- **Total training cost**: ~150 GPU-hours end-to-end across all three stages

The Memory Manager and Retrieve Agent are trained as one-shot JSON generation problems with rule-based reward. The Answer Agent produces short answers conditioned on retrieved memories. During Memory Manager and Retrieve Agent training, the Answer Agent is frozen and used only for reward computation.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **F1** | Token-level F1 between predicted and gold answers (lexical overlap) |
| **B1** | BLEU-1 unigram precision (answer conciseness + correctness) |
| **J** | LLM-as-judge accuracy (GPT-4o mini evaluates semantic correctness) |

## Baseline: Memory-R1

`memory_r1/` contains our re-implementation of **Memory-R1**, a flat-memory system with an RL-trained Memory Manager and Answer Agent. Memory-R1 does not provide a public codebase; we re-implement it from scratch based on the original paper and retrain the backbone on the same base model and data split for fair comparison. This serves as both our primary RL-based baseline and a component of our ablation study. The Answer Agent trained here is reused by StructMemoryR1 as a shared component across both systems.

See [memory_r1/README.md](memory_r1/README.md) for training instructions.

```bash
bash memory_r1/scripts/nebius_train.sh --stage both
```

## Repository Structure

```
.
├── struct_memory_r1/              # Main contribution: 3-agent RL system
│   ├── agents/                    #   Answer Agent, Retrieve Agent, Memory Manager
│   ├── data/                      #   LoCoMo data loading + parquet builder
│   ├── memory_bank.py             #   Tree-structured memory bank T=(V,E)
│   ├── prompts.py                 #   Prompt templates for all 3 agents
│   ├── reward/                    #   Reward functions (EM, F1, evidence hit)
│   ├── retrieval/                 #   Memory tree retrieval server
│   └── scripts/                   #   Training scripts (per-agent + unified)
│       ├── train_answer_agent.sh      # Stage 1: Answer Agent
│       ├── train_retrieve_agent.sh    # Stage 2: Retrieve Agent
│       ├── train_memory_manager.sh    # Stage 3: Memory Manager
│       └── nebius_train.sh            # All 3 stages (cloud GPU)
├── memory_r1/                     # Baseline: flat memory (2-agent)
│   ├── data/                      #   LoCoMo data loading + parquet builder
│   └── scripts/                   #   Baseline training scripts
├── structured_locomo_trees/       # Struct-LoCoMo: tree-structured memory XMLs
├── backbone/                      # Search-R1 infrastructure
├── verl/                          # RL training framework (GRPO/PPO via Ray+FSDP)
├── search_r1/                     # Retrieval backbone
├── checkpoints/                   # Trained model weights (see checkpoints/README.md)
├── data/locomo/                   # Raw LoCoMo dataset
└── figures/                       # Paper figures
```

## Quick Start

### 1. Install Dependencies

```bash
conda create -n structmemory python=3.10 -y
conda activate structmemory
pip install -e .
pip install vllm==0.6.3 ray==2.10.0 flash-attn --no-build-isolation
```

### 2. Build Training Data

```bash
python -m struct_memory_r1.data.build_training_data
```

### 3. Start Memory Retrieval Server

```bash
python -m struct_memory_r1.retrieval.memory_server
```

### 4. Train

**Stage 1 — Answer Agent:**
```bash
bash struct_memory_r1/scripts/train_answer_agent.sh
```

**Stage 2 — Retrieve Agent (frozen Answer Agent):**
```bash
FROZEN_ANSWER_AGENT=verl_checkpoints/<stage1-ckpt> \
bash struct_memory_r1/scripts/train_retrieve_agent.sh
```

**Stage 3 — Memory Manager (frozen Answer Agent + Retrieve Agent):**
```bash
FROZEN_ANSWER_AGENT=verl_checkpoints/<stage1-ckpt> \
FROZEN_RETRIEVER=verl_checkpoints/<stage2-ckpt> \
bash struct_memory_r1/scripts/train_memory_manager.sh
```

**Full pipeline on a cloud GPU (e.g. Nebius H200):**
```bash
bash struct_memory_r1/scripts/nebius_train.sh --stage all
```

## Backbone

`verl/` and `search_r1/` are the underlying RL training framework and retrieval infrastructure, adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1) / [veRL](https://github.com/volcengine/verl). See [backbone/README.md](backbone/README.md) for details.

## License

See [LICENSE](LICENSE).
