# Memory-R1 Architecture & Codebase Guide

Complete technical reference for the Memory-R1 implementation. Maps every paper concept to its code location.

---

## Directory Structure

```
memory_r1/
├── __init__.py
├── prompts.py                    # All prompts (Figures 9-11, Figure 12)
├── memory_bank.py                # Memory bank data structure + CRUD operations
├── llm_extract.py                # LLM-based fact extraction via OpenAI API
├── agents/
│   ├── __init__.py
│   └── memory_manager.py         # Memory Manager agent, frozen Answer Agent, reward functions
├── data/
│   ├── __init__.py
│   ├── locomo_loader.py          # LoCoMo dataset download, parsing, and splitting
│   └── build_training_data.py    # Algorithms 1 & 2: training data construction
├── reward/
│   ├── __init__.py
│   └── em_reward.py              # EM reward, answer extraction, score computation
├── retrieval/
│   ├── __init__.py
│   └── memory_server.py          # FastAPI retrieval server (TF-IDF based)
└── tests/
    ├── __init__.py
    ├── test_memory_bank.py       # 22 tests: CRUD, parsing, paper examples
    ├── test_prompts.py           # 9 tests: prompt content, formatting, examples
    ├── test_data.py              # 23 tests: parsing, extraction, retrieval, data format
    ├── test_reward.py            # 28 tests: EM, answer extraction, reward functions
    └── test_verl_integration.py  # 24 tests: routing, frozen agent, parquet format

train_answer_agent.sh             # Stage 1 training script
train_memory_manager.sh           # Stage 2 training script
verl/trainer/main_ppo.py          # veRL reward routing (modified for Memory-R1)
```

---

## Paper-to-Code Mapping

### Section 3.1: Memory Manager

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| Policy π_θ (Eq. 1) | Trained via veRL GRPO | The Memory Manager LLM itself |
| Operations {ADD, UPDATE, DELETE, NOOP} | `memory_bank.py:113-141` | `MemoryBank.apply_operations()` |
| Memory Manager prompt (Figures 9-10) | `prompts.py:16-102` | `MEMORY_MANAGER_SYSTEM` constant |
| Format prompt with turn context | `prompts.py:207-233` | `make_memory_manager_training_prompt()` |
| R_answer = EM(y_pred, y_gold) (Eq. 4) | `agents/memory_manager.py:305-390` | `compute_score_memory_manager_verl()` |
| Frozen Answer Agent | `agents/memory_manager.py:57-163` | `FrozenAnswerAgent` class |
| Global frozen agent state | `agents/memory_manager.py:166-178` | `set_frozen_answer_agent()` / `get_frozen_answer_agent()` |
| GRPO training | `train_memory_manager.sh` | Shell script with veRL config |

### Section 3.2: Answer Agent

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| Policy π_θ (Eq. 5) | Trained via veRL GRPO | The Answer Agent LLM itself |
| 60 candidate memories (30 per speaker) | `data/build_training_data.py:396-496` | `build_answer_agent_data()` with `memories_per_speaker=30` |
| Memory Distillation | `prompts.py:134-160` | `ANSWER_AGENT_SYSTEM` instruction #9 |
| Speaker-grouped prompt (Figure 11) | `prompts.py:163-200` | `format_answer_agent_prompt()` |
| R = EM(y_pred, y_gold) | `reward/em_reward.py:96-124` | `compute_score_answer_agent()` |
| Answer extraction (`**Answer:**`) | `reward/em_reward.py:71-93` | `extract_answer_from_output()` |
| GRPO training | `train_answer_agent.sh` | Shell script with veRL config |

### Section 4.1: Dataset & Evaluation

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| LoCoMo loading | `data/locomo_loader.py:52-224` | Download, parse, split |
| Adversarial filtering | `data/locomo_loader.py:164` | `if category == "adversarial": continue` |
| 1:1:8 split (152/81/1307) | `data/locomo_loader.py:200-218` | `split_locomo_train_val_test()` |
| EM normalization | `reward/em_reward.py:18-36` | `normalize_answer()` |
| LLM-as-a-Judge prompt (Figure 12) | `prompts.py:279-304` | `JUDGE_PROMPT_TEMPLATE` |

### Appendix B.2: Training Data Construction

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| Algorithm 1 (MM data, per-turn) | `data/build_training_data.py:277-389` | `build_memory_manager_data()` |
| GPT-4o-mini fact extraction | `llm_extract.py:58-133` | `llm_extract_facts()` (uses GPT-5-nano) |
| Heuristic fallback extraction | `data/build_training_data.py:43-61` | `extract_facts_from_turn_heuristic()` |
| Temporal memory bank (50 turns) | `data/build_training_data.py:113-146` | `build_temporal_memory_bank_from_turns()` |
| QA pair linking to turns | `data/build_training_data.py:168-205` | `get_qa_pairs_up_to_turn()` |
| Algorithm 2 (AA data, retrieval) | `data/build_training_data.py:396-496` | `build_answer_agent_data()` |
| TF-IDF retrieval (approx RAG) | `data/build_training_data.py:233-270` | `tfidf_retrieve()` |

### Appendix E: Algorithms 3-5

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| Algorithm 3 (Memory Bank Construction) | `memory_bank.py:113-141` | `MemoryBank.apply_operations()` |
| Algorithm 4 (Answer Generation) | `agents/memory_manager.py:91-129` | `FrozenAnswerAgent.answer()` |
| Algorithm 5 (MM Training Pipeline) | `agents/memory_manager.py:305-390` | `compute_score_memory_manager_verl()` |
| Algorithm 5: old_memory threading | `verl/trainer/main_ppo.py:84-91` | `extra_info` passed to reward fn |

### Appendix D: Implementation Details

| Paper Concept | Code Location | Description |
|---------------|---------------|-------------|
| veRL/GRPO framework | `verl/trainer/main_ppo.py` | Modified for Memory-R1 data sources |
| Reward routing | `verl/trainer/main_ppo.py:25-35` | `_select_rm_score_fn()` |
| Frozen agent setup | `verl/trainer/main_ppo.py:189-208` | Loads model at training start |
| τ=1.0 for training | Training scripts | `actor_rollout_ref.rollout.temperature=1` |
| Single-turn agents | Training scripts | `do_search=false`, `max_turns=1` |

---

## Component Deep Dives

### Memory Bank (`memory_bank.py`)

The core data structure. A flat list of `{id, text}` entries supporting:

```python
bank = MemoryBank()
bank.add("User likes pizza")              # → MemoryEntry(id="0", text="User likes pizza")
bank.add("Name is John", entry_id="1")    # → MemoryEntry(id="1", text="Name is John")
bank.update("0", "User loves pepperoni")  # Updates id=0's text
bank.delete("1")                          # Removes id=1
bank.to_list()                            # → [{"id": "0", "text": "User loves pepperoni"}]
```

**Parsing Memory Manager output** (`parse_memory_manager_output`):
- Extracts JSON from model output (handles markdown code blocks, extra text)
- Looks for `{"memory": [...]}` structure
- Returns `(operations, success)` tuple

**Applying operations** (`apply_operations`):
- Creates a deep copy of the bank
- Processes each operation by event type
- UPDATE on nonexistent ID falls back to ADD (graceful handling)
- Returns a new `MemoryBank` (immutable pattern)

### Prompts (`prompts.py`)

Three prompt templates, each reproduced verbatim from the paper:

**`MEMORY_MANAGER_SYSTEM`** (Figures 9-10):
- 4 detailed examples with exact JSON format
- ADD: "User is a software engineer" + "Name is John"
- UPDATE: "cheese pizza" → "cheese and chicken pizza" with `old_memory` field
- DELETE: "Loves cheese pizza" with "Dislikes cheese pizza" contradiction
- NONE: "Name is John" already present

**`ANSWER_AGENT_SYSTEM`** (Figure 11):
- 10 numbered instructions (timestamps, character names, 5-6 words, memory selection)
- 8-step reasoning approach
- Memory Distillation: "Select memories you found that are useful"

**`JUDGE_PROMPT_TEMPLATE`** (Figure 12):
- CORRECT/WRONG labeling with generous grading
- Handles time-related questions with format flexibility

**Formatting functions:**
- `format_memory_manager_prompt(old_memory, retrieved_facts)` — produces the input section
- `format_answer_agent_prompt(question, memories_a, memories_b, ...)` — speaker-grouped format with count annotations
- `make_*_training_prompt(...)` — combines system prompt + formatted input for veRL

### Data Pipeline (`data/`)

#### LoCoMo Loader (`locomo_loader.py`)

Downloads and parses `locomo10.json` from GitHub:

```python
conversations = load_locomo("data/locomo")  # Downloads if not present
# → List of 10 Conversation objects
#   Each has: sample_id, speaker_a, speaker_b, sessions[], qa_pairs[]
#   Each Session has: turns[], observations[], datetime, summary
#   Each QAPair has: question, answer, category, evidence[]
```

Key parsing details:
- Speaker names extracted from first session's dialogue turns (not from top-level keys)
- QA categories mapped from integers: `{1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}`
- Adversarial questions filtered out
- Evidence IDs in `D5:3` format (session 5, turn 3)
- All answers coerced to strings (some LoCoMo answers are integers)

#### Training Data Builder (`build_training_data.py`)

**Memory Manager data** (`build_memory_manager_data`, Algorithm 1):

For each dialogue turn `t` in each conversation:
1. Collect all turns across all sessions with session metadata
2. Skip turns with <10 characters (greetings)
3. Build temporal memory bank from previous turns:
   - With `--use_llm`: extract facts from each of the previous 50 turns via OpenAI API
   - Without `--use_llm`: use LoCoMo observations from previous sessions
4. Extract facts from the current turn
5. Link QA pairs whose evidence falls at or before this turn
6. Package as `{data_source, prompt, reward_model, extra_info}` for veRL

**Answer Agent data** (`build_answer_agent_data`, Algorithm 2):

For each QA pair in each conversation:
1. Build observation pools per speaker from all sessions (with timestamps)
2. Attribute each observation to a speaker by name mention
3. For each question, retrieve top-30 per speaker via TF-IDF similarity
4. Format with speaker grouping, timestamps, and Memory Distillation instruction
5. Package as veRL training sample with gold answer as target

### Fact Extraction (`llm_extract.py`)

LLM-based extraction via OpenAI API (paper uses GPT-4o-mini, we use GPT-5-nano):

```python
from memory_r1.llm_extract import llm_extract_facts

facts = llm_extract_facts(
    turn_text="I just adopted a dog named Buddy from the shelter.",
    speaker="Alice",
    model="gpt-5-nano",
)
# → ["Alice adopted a dog named Buddy from the shelter."]
```

Features:
- System prompt instructs extraction of concrete facts, events, preferences
- Handles markdown code blocks in model output
- Retry with exponential backoff on API failures
- Lazy OpenAI client initialization (only when called)

### Reward Functions (`reward/em_reward.py`)

**Answer normalization** (`normalize_answer`):
- Lowercase → remove punctuation → remove articles (a, an, the) → whitespace fix

**EM check** (`em_check`):
- Compares normalized prediction against any gold answer
- Returns 1 (match) or 0 (no match)

**Sub-EM check** (`subem_check`):
- Gold answer is substring of prediction (more lenient)

**Answer extraction** (`extract_answer_from_output`):
- Tries `**Answer:** <text>` pattern first
- Falls back to `<answer>...</answer>` tags (Search-R1 compatibility)
- Last resort: returns last non-empty line

**Score functions:**
- `compute_score_answer_agent(solution_str, ground_truth)` — direct EM on extracted answer
- `compute_score_memory_manager(solution_str, ground_truth)` — format validation only
- `compute_score_memory_r1(solution_str, ground_truth)` — EM for Memory-R1 Answer Agent format
- `compute_score_memory_manager_verl(solution_str, ground_truth, extra_info)` — full frozen agent reward

### Memory Manager Agent (`agents/memory_manager.py`)

**`FrozenAnswerAgent`** class:
- Wraps a frozen Answer Agent model for computing Memory Manager reward
- Supports two inference backends:
  - API endpoint (`api_url`): sends prompt to a vLLM/API server
  - Local model (`model_path`): loads HuggingFace model for in-process inference
- `answer(question, memories)` → builds prompt with `ANSWER_AGENT_SYSTEM`, generates answer

**Global state management:**
```python
set_frozen_answer_agent(agent)   # Called at training start
get_frozen_answer_agent()        # Called during reward computation
```

**`compute_score_memory_manager_verl`** (Algorithm 5 reward):
1. Parse model output → memory operations
2. Validate event types (ADD/UPDATE/DELETE/NONE/NOOP)
3. Reconstruct old memory bank from `extra_info.old_memory`
4. Apply operations → new memory bank
5. Run frozen Answer Agent on each QA pair with new memories
6. Return mean EM score across QA pairs

### Retrieval Server (`retrieval/memory_server.py`)

FastAPI server providing Search-R1-compatible retrieval API:

```python
# MemoryStore: TF-IDF indexed memory bank
store = MemoryStore()
store.load_memories([{"text": "Alice has a dog", "speaker": "Alice"}, ...])
results = store.retrieve("What pet does Alice have?", topk=30, speaker="Alice")
```

API endpoints:
- `POST /retrieve` — batch query retrieval, returns documents or scored results
- `POST /update_memories` — replace the memory bank
- `GET /status` — returns `{num_memories: N}`

### veRL Integration (`verl/trainer/main_ppo.py`)

**Reward routing** (`_select_rm_score_fn`):
```python
"nq", "triviaqa", ...  → qa_em.compute_score_em        # Search-R1 datasets
"answer_agent"          → compute_score_memory_r1        # Memory-R1 Answer Agent
"memory_manager"        → compute_score_memory_manager_verl  # Memory-R1 Memory Manager
```

**RewardManager modifications:**
- For `memory_manager` data source, passes `extra_info` from parquet to the reward function
- This enables the frozen Answer Agent to use the correct old memory bank for UPDATE/DELETE operations

**Frozen Answer Agent setup** (in `main_task`):
- Checks for `frozen_answer_agent_path` or `frozen_answer_agent_url` config
- Loads model/tokenizer and calls `set_frozen_answer_agent()`
- Activated via `FROZEN_ANSWER_AGENT` env var in `train_memory_manager.sh`

---

## Data Flow Diagrams

### Training Data Construction

```
LoCoMo JSON (10 conversations, ~600 turns, ~1540 QA pairs)
    │
    ├─→ Filter adversarial QAs (removes ~category 5)
    │
    ├─→ Split 1:1:8 by conversations (1 train, 1 val, 8 test)
    │
    ├─→ Memory Manager Pipeline (per-turn):
    │     For each turn t:
    │       previous turns → [fact extraction] → temporal memory bank
    │       current turn   → [fact extraction] → retrieved facts
    │       QA pairs       → [evidence linking] → linked QAs
    │       ────────────────────────────────────────────────
    │       Output: {prompt, old_memory, facts, qa_targets}
    │
    └─→ Answer Agent Pipeline (per-QA):
          For each QA pair:
            all observations → [speaker attribution] → per-speaker pools
            question         → [TF-IDF retrieval]    → 30 per speaker
            ────────────────────────────────────────────────
            Output: {prompt (speaker-grouped), gold_answer}
```

### Training Reward Flow

```
Answer Agent Training:
  prompt → [LLM generates response] → extract "**Answer:**" → EM(pred, gold) → reward

Memory Manager Training (with frozen Answer Agent):
  prompt → [LLM generates JSON operations] → parse operations
       → reconstruct old_memory from extra_info
       → apply operations → new memory bank
       → [frozen Answer Agent](question, new_memories) → predicted answer
       → EM(predicted, gold) → reward
```

---

## Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_memory_bank.py` | 22 | CRUD operations, copy, serialization, parsing, paper examples |
| `test_prompts.py` | 9 | Prompt content verification, formatting, speaker grouping |
| `test_data.py` | 23 | LoCoMo parsing, fact extraction, temporal bank, QA linking, TF-IDF, veRL format |
| `test_reward.py` | 28 | Normalization, EM/sub-EM, answer extraction, all score functions |
| `test_verl_integration.py` | 24 | Reward routing, frozen agent, extra_info threading, parquet format, data counts |
| **Total** | **106** | |

Run all tests:
```bash
conda run --no-capture-output -n searchr1 python -m pytest memory_r1/tests/ -v
```

---

## Assumptions & Deviations from Paper

| Area | Paper | Implementation | Reason |
|------|-------|---------------|--------|
| Turn window | Algorithm 1 says "50 turns"; Section B.2 text says "24 turns" | Default: 50 (configurable) | Algorithm pseudocode is the formal spec |
| Fact extraction | GPT-4o-mini | GPT-5-nano (or heuristic fallback) | User preference; heuristic available without API |
| Retrieval | "similarity-based RAG" (dense embeddings implied) | TF-IDF cosine similarity | Sufficient for pipeline; swappable |
| Prompt/response lengths | 4096/2048 tokens | 2048/256-512 tokens | 24GB VRAM constraint |
| Batch size | 128 (8 GPUs) | 8 (1 GPU) | Hardware constraint |
| Algorithm 5 online loop | Sequential turn processing with online RL updates | Static per-turn data + frozen agent reward | Would require deep veRL modifications |
| AA training data memories | Built by Memory Manager | Built from LoCoMo observations | MM not trained during data construction (chicken-and-egg) |
