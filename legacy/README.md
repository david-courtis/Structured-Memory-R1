# Legacy / unused

This directory contains code that is **not part of the StructMemoryR1 or
Memory-R1 training pipelines** and is preserved here only for historical
reference. Nothing under `legacy/` is imported by the active packages
(`struct_memory_r1/`, `memory_r1/`, `verl/`, `search_r1/`).

## Contents

- **`struct_memory_r1_retriever/`** — Early fork of the structured-memory
  retriever package. It was merged into `struct_memory_r1/`; the surviving
  files are shims (`__init__.py`, `agents/retriever_agent.py`) plus stale
  copies of `memory_bank.py`, `prompts.py`, and `agents/memory_manager.py`.
  Two smoke scripts that used to live under `struct_memory_r1/scripts/`
  referenced this package and were removed alongside the move.

- **`backbone/`** — Scaffolding inherited from
  [Search-R1](https://github.com/PeterGriffinJin/Search-R1): example configs,
  multi-node launch scripts, NQ/HotpotQA training shells, and a sample corpus.
  The active `verl/` and `search_r1/` Python packages at the repository root
  are what training actually uses.

- **`legacy_search_r1_scripts/`** — Top-level `scripts/` from the original
  Search-R1 repo: NQ/HotpotQA data processing, wandb upload helpers, and
  training-curve plotting. None are invoked by the LoCoMo training pipelines.

If you need to revive any of this, it is checked into git history under its
original path; `git log --follow` will trace it.
