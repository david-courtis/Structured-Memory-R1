# StructMemoryR1

StructMemoryR1 instantiates a memory-augmented LLM framework with **three learned components** trained end-to-end with GRPO using downstream question-answering reward:

| Component | Symbol | Role |
|-----------|--------|------|
| Memory Manager | π_θM | Induces and updates tree-structured memory via CRUD ops |
| Retrieve Agent | π_θR | Plans a structured search over the memory tree schema |
| Answer Agent | π_θA | Answers questions from retrieved evidence |

Training is **staged**: Answer Agent first (then frozen), then Retrieve Agent, then Memory Manager. This ensures stable reward signals at each phase.

## Memory Bank: Tree Structure

The memory bank `T = (V, E)` is a rooted tree constructed incrementally from raw conversation:

```
Level 0:   root  (full conversation)
Level 1:   speakers  (e.g. "Calvin", "Dave")
Level 2:   topics    (e.g. "Music", "Cars", "Travel")
Level 3:   facts     (leaf nodes — individual memory entries)
```

Each internal node carries a **summary** `s(v)` that aggregates its children. The **schema** of the tree consists of node IDs and summaries at each level without exposing full leaf content — providing a compact representation for the Retrieve Agent.

The tree is serialized as structured JSON for LLM input.

## Module Structure

```
struct_memory_r1/
├── memory_bank.py              # Tree T=(V,E), CRUD executor Apply(T, o_t),
│                               #   and retrieval executor Execute(T, z)
├── prompts.py                  # Prompt templates for all 3 agents
│                               #   MEMORY_MANAGER_SYSTEM, RETRIEVER_AGENT_SYSTEM,
│                               #   ANSWER_AGENT_SYSTEM
├── llm_extract.py              # LLM-based fact extraction from raw dialogue
├── agents/
│   ├── memory_manager.py       # π_θM: CRUD ops on tree + reward r_M
│   ├── retrieve_agent.py       # π_θR: search plan over schema + reward r_R
│   └── (answer_agent implicit  # π_θA: trained in Stage 1; frozen in Stages 2-3)
├── reward/
│   └── em_reward.py            # Shared: EM, token F1, answer extraction
├── retrieval/
│   └── memory_server.py        # FastAPI server for search plan execution
└── tests/
```

## Agents

### Memory Manager (`agents/memory_manager.py`)

Given dialogue turn `x_t` and current tree `T_t`, samples an operation sequence:

```
o_t ~ π_θM(· | x_t, T_t)
T_{t+1} = Apply(T_t, o_t)
```

Operations are restricted to `{ADD, UPDATE, DELETE, NONE}`. Output is a JSON object specifying entries to add/modify/remove and their placement in the tree hierarchy.

**Reward:**
```
r_M = R(y_t, y_t*) + λ_f · 1[valid JSON]
```
where `R(y, y*)` is token F1 of the downstream Answer Agent output on the updated tree.

### Retrieve Agent (`agents/retrieve_agent.py`)

Given question `q` and tree schema `Schema(T)`, emits a search plan:

```
z ~ π_θR(· | q, Schema(T))
R = Execute(T, z)
```

The search plan `z` is JSON with:
- `levels`: branch keys to expand at each tree level
- `selected_ids`: specific leaf nodes to retrieve directly
- `stop`: terminates search

**Reward:**
```
r_R = R(y_t, y_t*) + λ_ev · Hit(R, E*) - λ_cost · C(z)
```
where `E*` are gold evidence nodes, `C(z)` penalises search cost.

### Answer Agent

Receives question `q` and retrieved memories `R`, produces a response:

```
y ~ π_θA(· | q, R)
```

Trained in Stage 1 with token F1 reward. Frozen in Stages 2 and 3 as the evaluator in the reward loop.

## Training Stages

```
Stage 1:  Train Answer Agent     (reward: token F1 vs gold answer)
Stage 2:  Train Retrieve Agent   (reward: F1 + evidence hit - cost; Answer Agent frozen)
Stage 3:  Train Memory Manager   (reward: F1 + JSON bonus; Retrieve Agent + Answer Agent frozen)
```

### Quick Start

```bash
# Stage 1 — Answer Agent
bash struct_memory_r1/scripts/train_answer_agent.sh

# Stage 2 — Retrieve Agent  (set FROZEN_ANSWER_AGENT path)
FROZEN_ANSWER_AGENT=verl_checkpoints/<stage1-ckpt> \
bash struct_memory_r1/scripts/train_retrieve_agent.sh

# Stage 3 — Memory Manager  (set FROZEN_ANSWER_AGENT + FROZEN_RETRIEVER)
FROZEN_ANSWER_AGENT=verl_checkpoints/<stage1-ckpt> \
FROZEN_RETRIEVER=verl_checkpoints/<stage2-ckpt> \
bash struct_memory_r1/scripts/train_memory_manager.sh

# Full pipeline on cloud GPU (e.g. Nebius H200)
bash struct_memory_r1/scripts/nebius_train.sh --stage all
```

## Tests

```bash
python -m pytest struct_memory_r1/tests/ -v
```
