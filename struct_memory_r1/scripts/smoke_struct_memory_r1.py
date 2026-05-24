#!/usr/bin/env python3
"""
Functional smoke test for struct_memory_r1.

This script is designed for machines like Apple Silicon Macs where the full
veRL + Ray + vLLM stack is typically unavailable. It validates the structured
memory pipeline end-to-end without running real training:

- environment feasibility check for full veRL smoke
- structured memory insertion / retrieval
- mini training-data construction
- reward computation with numpy-style parquet targets
"""

import importlib
import platform
import sys

import numpy as np

from struct_memory_r1.agents.memory_manager import (
    FrozenAnswerAgent,
    compute_score_memory_manager_verl,
    set_frozen_answer_agent,
)
from struct_memory_r1.data.build_training_data import (
    build_answer_agent_data,
    build_memory_manager_data,
)
from struct_memory_r1.data.locomo_loader import Conversation, DialogueTurn, QAPair, Session
from struct_memory_r1.memory_bank import MemoryBank, parse_memory_manager_output


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def full_verl_smoke_feasible() -> tuple[bool, str]:
    system = platform.system()
    machine = platform.machine().lower()

    required = ["ray", "vllm", "xformers"]
    missing = [name for name in required if not module_available(name)]

    if system != "Linux":
        return False, f"host is {system}, but the current veRL smoke script is CUDA/Linux-oriented"
    if "arm" in machine and missing:
        return False, f"missing runtime pieces: {', '.join(missing)}"
    if missing:
        return False, f"missing runtime pieces: {', '.join(missing)}"
    return True, "full veRL smoke prerequisites detected"


def make_test_conversation() -> Conversation:
    sessions = [
        Session(
            session_id=1,
            datetime="2023-01-15 10:00",
            turns=[
                DialogueTurn(
                    speaker="Alice",
                    dia_id="D1:1",
                    text="Hi Bob! I just adopted a dog named Buddy from the shelter.",
                ),
                DialogueTurn(
                    speaker="Bob",
                    dia_id="D1:2",
                    text="That is great Alice! What breed is he?",
                ),
                DialogueTurn(
                    speaker="Alice",
                    dia_id="D1:3",
                    text="He is a golden retriever, about 2 years old.",
                ),
            ],
            observations=[
                "Alice adopted a dog named Buddy from the shelter.",
                "Buddy is a golden retriever, about 2 years old.",
            ],
            summary="Alice tells Bob about her new dog Buddy.",
        ),
        Session(
            session_id=2,
            datetime="2023-03-20 14:00",
            turns=[
                DialogueTurn(
                    speaker="Alice",
                    dia_id="D2:1",
                    text="I ended up adopting another dog. His name is Scout.",
                ),
                DialogueTurn(
                    speaker="Bob",
                    dia_id="D2:2",
                    text="Two dogs is wonderful.",
                ),
            ],
            observations=[
                "Alice adopted another dog named Scout.",
                "Alice now has two dogs: Buddy and Scout.",
            ],
            summary="Alice adopts a second dog named Scout.",
        ),
    ]

    qa_pairs = [
        QAPair(
            question="How many dogs does Alice have?",
            answer="2",
            category="multi-hop",
            evidence=["D2:1"],
        ),
        QAPair(
            question="What is Alice's first dog's name?",
            answer="Buddy",
            category="single-hop",
            evidence=["D1:1"],
        ),
    ]

    return Conversation(
        sample_id="struct_smoke",
        speaker_a="Alice",
        speaker_b="Bob",
        sessions=sessions,
        qa_pairs=qa_pairs,
    )


class MockAgent(FrozenAnswerAgent):
    def answer(self, question, memories):
        text_blob = " ".join(m.get("text", "") for m in memories)
        if "How many dogs" in question and "Buddy" in text_blob and "Scout" in text_blob:
            return "**Answer:** 2"
        if "first dog's name" in question and "Buddy" in text_blob:
            return "**Answer:** Buddy"
        return "**Answer:** unknown"


def run_functional_smoke() -> list[str]:
    conv = make_test_conversation()

    bank = MemoryBank()
    bank.insert_interaction("Alice: I adopted a dog named Buddy.", timestamp="2023-01-15 10:00")
    bank.insert_interaction("Alice: I later adopted another dog named Scout.", timestamp="2023-03-20 14:00")
    retrieval = bank.retrieve("How many dogs does Alice have?", speaker="Alice", topk=2)
    if len(retrieval) < 2:
        raise AssertionError("structured retrieval did not return the expected dog memories")

    ops, ok = parse_memory_manager_output(
        '{"structured_memory": [{"id": "0", "text": "Alice adopted Buddy", "event": "ADD", "path": ["Alice", "dog"]}]}'
    )
    if not ok or ops[0]["path"] != ["Alice", "dog"]:
        raise AssertionError("structured memory parser did not accept structured_memory payload")

    restructured = bank.apply_operations([
        {"event": "CREATE_SUBTOPIC", "path": ["Alice", "pets", "dogs"], "parent_path": ["Alice", "pets"], "text": "dogs"},
        {"id": "0", "event": "MOVE", "path": ["Alice", "pets", "dogs"]},
        {"id": "1", "event": "MOVE", "path": ["Alice", "pets", "dogs"]},
    ])
    moved_paths = [entry["path"] for entry in restructured.to_structured_list()]
    if ["Alice", "pets", "dogs"] not in moved_paths:
        raise AssertionError("structure-edit operations did not move facts into the new subtopic")

    mm_samples = build_memory_manager_data([conv], split="train")
    aa_samples = build_answer_agent_data([conv], split="test")

    if not mm_samples or mm_samples[0]["data_source"] != "struct_memory_manager":
        raise AssertionError("struct memory manager data was not built correctly")
    if "Structured Memory Tree" not in mm_samples[0]["prompt"][0]["content"]:
        raise AssertionError("struct memory manager prompt is missing the tree view")
    if not aa_samples or aa_samples[0]["data_source"] != "struct_answer_agent":
        raise AssertionError("struct answer agent data was not built correctly")

    set_frozen_answer_agent(MockAgent())
    try:
        score = compute_score_memory_manager_verl(
            solution_str=(
                '{"memory": ['
                '{"id": "0", "text": "Alice adopted a dog named Buddy", "event": "ADD", "speaker": "Alice", "path": ["Alice", "dog"]},'
                '{"id": "1", "text": "Alice adopted another dog named Scout", "event": "ADD", "speaker": "Alice", "path": ["Alice", "dog"]}'
                ']}'
            ),
            ground_truth={
                "target": np.array(
                    [{"question": "How many dogs does Alice have?", "answer": "2"}],
                    dtype=object,
                )
            },
        )
    finally:
        set_frozen_answer_agent(None)

    if score <= 0.0:
        raise AssertionError(f"expected a positive reward from structured smoke test, got {score}")

    return [
        f"retrieval_hits={len(retrieval)}",
        f"memory_manager_samples={len(mm_samples)}",
        f"answer_agent_samples={len(aa_samples)}",
        f"reward_score={score}",
    ]


def main() -> int:
    feasible, reason = full_verl_smoke_feasible()
    print(f"full_verl_smoke_feasible={feasible}")
    print(f"full_verl_smoke_reason={reason}")

    details = run_functional_smoke()
    print("functional_struct_smoke=PASS")
    for item in details:
        print(item)

    if not feasible:
        print("recommendation=use this functional smoke on Apple Silicon; do not expect the CUDA veRL smoke script to run here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
