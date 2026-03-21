"""Tests for LoCoMo data loading and training data construction."""
import os
import pytest
from unittest.mock import patch

from memory_r1.data.locomo_loader import (
    DialogueTurn, Session, QAPair, Conversation,
    parse_locomo, get_all_turns_flat, get_all_observations,
    split_locomo_train_val_test,
)
from memory_r1.data.build_training_data import (
    extract_facts_from_turn_heuristic,
    extract_facts_from_turn,
    extract_facts_from_observations,
    build_temporal_memory_bank,
    build_temporal_memory_bank_from_turns,
    build_memory_manager_data,
    build_answer_agent_data,
    tfidf_retrieve,
    get_qa_pairs_up_to_turn,
)


def make_test_conversation():
    """Create a minimal test conversation."""
    sessions = [
        Session(
            session_id=1,
            datetime="2023-01-15 10:00",
            turns=[
                DialogueTurn(speaker="Alice", dia_id="D1:1",
                             text="Hi Bob! I just adopted a dog named Buddy from the shelter."),
                DialogueTurn(speaker="Bob", dia_id="D1:2",
                             text="That's great Alice! What breed is he?"),
                DialogueTurn(speaker="Alice", dia_id="D1:3",
                             text="He's a golden retriever, about 2 years old."),
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
                DialogueTurn(speaker="Alice", dia_id="D2:1",
                             text="I can't help but adopt another dog. His name is Scout."),
                DialogueTurn(speaker="Bob", dia_id="D2:2",
                             text="Two dogs! That's wonderful."),
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
        sample_id="test_1",
        speaker_a="Alice",
        speaker_b="Bob",
        sessions=sessions,
        qa_pairs=qa_pairs,
    )


class TestDialogueStructures:

    def test_conversation_creation(self):
        conv = make_test_conversation()
        assert conv.sample_id == "test_1"
        assert len(conv.sessions) == 2
        assert len(conv.qa_pairs) == 2

    def test_get_all_turns(self):
        conv = make_test_conversation()
        turns = get_all_turns_flat(conv)
        assert len(turns) == 5
        assert turns[0].speaker == "Alice"
        assert turns[-1].speaker == "Bob"

    def test_get_all_observations(self):
        conv = make_test_conversation()
        obs = get_all_observations(conv)
        assert len(obs) == 4
        assert "Buddy" in obs[0]

    def test_split(self):
        convs = [make_test_conversation() for _ in range(10)]
        train, val, test = split_locomo_train_val_test(convs)
        assert len(train) >= 1
        assert len(val) >= 1
        assert len(test) >= 1
        assert len(train) + len(val) + len(test) == 10


class TestFactExtraction:

    def test_extract_from_turn_heuristic(self):
        turn = DialogueTurn(
            speaker="Alice", dia_id="D1:1",
            text="I just adopted a dog named Buddy from the shelter. He's really cute!"
        )
        facts = extract_facts_from_turn_heuristic(turn)
        assert len(facts) >= 1
        assert any("Buddy" in f for f in facts)

    def test_extract_short_turn_filtered(self):
        turn = DialogueTurn(speaker="Bob", dia_id="D1:2", text="Cool!")
        facts = extract_facts_from_turn_heuristic(turn)
        assert len(facts) == 0

    def test_extract_from_observations(self):
        conv = make_test_conversation()
        facts = extract_facts_from_observations(conv, session_idx=0)
        assert len(facts) == 2
        assert "Buddy" in facts[0]

    def test_extract_from_turn_without_llm(self):
        """Test that extract_facts_from_turn works without LLM (default)."""
        turn = DialogueTurn(
            speaker="Alice", dia_id="D1:1",
            text="I just adopted a dog named Buddy from the shelter."
        )
        facts = extract_facts_from_turn(turn, use_llm=False)
        assert len(facts) >= 1


class TestTemporalMemoryBank:

    def test_empty_at_start(self):
        conv = make_test_conversation()
        bank = build_temporal_memory_bank(conv, up_to_session=0)
        assert len(bank) == 0

    def test_includes_first_session(self):
        conv = make_test_conversation()
        bank = build_temporal_memory_bank(conv, up_to_session=1)
        assert len(bank) == 2  # 2 observations from session 1

    def test_includes_both_sessions(self):
        conv = make_test_conversation()
        bank = build_temporal_memory_bank(conv, up_to_session=2)
        assert len(bank) == 4  # 2 + 2 observations

    def test_from_turns(self):
        """Test building memory bank from individual turns."""
        turns = [
            DialogueTurn(speaker="Alice", dia_id="D1:1",
                         text="I just adopted a dog named Buddy from the shelter."),
            DialogueTurn(speaker="Bob", dia_id="D1:2",
                         text="That's great Alice! What breed is he?"),
        ]
        bank, facts = build_temporal_memory_bank_from_turns(turns, use_llm=False)
        assert len(facts) >= 1
        assert len(bank) >= 1


class TestQAPairLinking:

    def test_qa_linked_to_turn(self):
        conv = make_test_conversation()
        # QA about Buddy is linked to D1:1 (session 1, turn 1)
        qas = get_qa_pairs_up_to_turn(conv, session_idx=1, turn_idx=1)
        assert any("Buddy" in qa.answer for qa in qas)

    def test_qa_not_linked_before_evidence(self):
        conv = make_test_conversation()
        # QA about "2 dogs" is linked to D2:1, shouldn't appear before session 2
        qas = get_qa_pairs_up_to_turn(conv, session_idx=1, turn_idx=3)
        assert not any(qa.answer == "2" for qa in qas)

    def test_qa_linked_after_evidence(self):
        conv = make_test_conversation()
        # QA about "2 dogs" should appear after D2:1
        qas = get_qa_pairs_up_to_turn(conv, session_idx=2, turn_idx=1)
        assert any(qa.answer == "2" for qa in qas)


class TestTfidfRetrieval:

    def test_retrieval_basic(self):
        candidates = [
            "Alice adopted a dog named Buddy",
            "Bob likes cats",
            "Alice went to the park",
            "Buddy is a golden retriever",
        ]
        results = tfidf_retrieve("What dog does Alice have?", candidates, topk=2)
        assert len(results) == 2
        # Should retrieve dog-related memories
        assert any("Buddy" in r or "dog" in r for r in results)

    def test_retrieval_empty(self):
        results = tfidf_retrieve("question", [], topk=5)
        assert results == []

    def test_retrieval_topk_limit(self):
        candidates = ["fact " + str(i) for i in range(100)]
        results = tfidf_retrieve("fact", candidates, topk=30)
        assert len(results) == 30


class TestBuildTrainingData:

    def test_memory_manager_data_per_turn(self):
        """Memory Manager data should have per-turn granularity."""
        conv = make_test_conversation()
        samples = build_memory_manager_data([conv], split="train")
        # Should have multiple samples per conversation (one per turn that has facts)
        assert len(samples) > 2  # More than per-session (which would be 2)

        sample = samples[0]
        assert sample["data_source"] == "memory_manager"
        assert "prompt" in sample
        assert sample["prompt"][0]["role"] == "user"
        assert "reward_model" in sample
        assert "turn_idx" in sample["extra_info"]

    def test_memory_manager_prompt_contains_examples(self):
        """The prompt should contain the full examples from Figures 9-10."""
        conv = make_test_conversation()
        samples = build_memory_manager_data([conv], split="train")
        content = samples[0]["prompt"][0]["content"]
        # Check for verbatim examples from the paper
        assert "User is a software engineer" in content
        assert "Loves cheese pizza" in content
        assert "old_memory" in content

    def test_answer_agent_data_speaker_grouped(self):
        """Answer Agent data should have speaker-grouped memories."""
        conv = make_test_conversation()
        samples = build_answer_agent_data([conv], split="test")
        assert len(samples) == 2  # 2 QA pairs

        sample = samples[0]
        assert sample["data_source"] == "answer_agent"
        content = sample["prompt"][0]["content"]
        # Should contain speaker-grouped sections
        assert "Memories for user Alice" in content
        assert "Memories for user Bob" in content
        # Should contain Memory Distillation instruction
        assert "Memories selected as relevant" in content

    def test_answer_agent_data_with_timestamps(self):
        """Answer Agent memories should include timestamps."""
        conv = make_test_conversation()
        samples = build_answer_agent_data([conv], split="test")
        content = samples[0]["prompt"][0]["content"]
        # Should contain timestamps from session datetimes
        assert "2023" in content

    def test_data_format_for_verl(self):
        """Verify the data format matches what veRL expects."""
        conv = make_test_conversation()
        samples = build_answer_agent_data([conv])

        for sample in samples:
            # veRL requires these fields
            assert "data_source" in sample
            assert "prompt" in sample
            assert isinstance(sample["prompt"], list)
            assert sample["prompt"][0]["role"] == "user"
            assert "content" in sample["prompt"][0]
            assert "reward_model" in sample
            assert "ground_truth" in sample["reward_model"]
            assert "target" in sample["reward_model"]["ground_truth"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
