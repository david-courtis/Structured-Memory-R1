"""Tests for Memory-R1 prompts."""
import pytest
from struct_memory_r1.prompts import (
    format_memory_manager_prompt,
    format_answer_agent_prompt,
    make_memory_manager_training_prompt,
    make_answer_agent_training_prompt,
    MEMORY_MANAGER_SYSTEM,
    ANSWER_AGENT_SYSTEM,
)


class TestMemoryManagerPrompt:

    def test_system_prompt_contains_operations(self):
        assert "ADD" in MEMORY_MANAGER_SYSTEM
        assert "UPDATE" in MEMORY_MANAGER_SYSTEM
        assert "DELETE" in MEMORY_MANAGER_SYSTEM
        assert "NONE" in MEMORY_MANAGER_SYSTEM

    def test_system_prompt_contains_examples(self):
        """Paper Figures 9-10: prompt must include detailed examples."""
        assert "User is a software engineer" in MEMORY_MANAGER_SYSTEM
        assert "Name is John" in MEMORY_MANAGER_SYSTEM
        assert "cheese pizza" in MEMORY_MANAGER_SYSTEM
        assert "play cricket" in MEMORY_MANAGER_SYSTEM
        assert "old_memory" in MEMORY_MANAGER_SYSTEM
        assert "speaker -> topic -> fact" in MEMORY_MANAGER_SYSTEM
        assert "CREATE_SUBTOPIC" in MEMORY_MANAGER_SYSTEM
        assert "SPLIT_TOPIC" in MEMORY_MANAGER_SYSTEM
        assert "MERGE_TOPIC" in MEMORY_MANAGER_SYSTEM

    def test_format_with_empty_memory(self):
        prompt = format_memory_manager_prompt(
            old_memory=[],
            retrieved_facts=["Name is John"],
        )
        assert "Old Memory:" in prompt
        assert "Retrieved facts:" in prompt
        assert "Name is John" in prompt

    def test_format_with_existing_memory(self):
        old = [{"id": "0", "text": "User is an engineer"}]
        prompt = format_memory_manager_prompt(
            old_memory=old,
            retrieved_facts=["User likes pizza"],
        )
        assert "engineer" in prompt
        assert "pizza" in prompt

    def test_training_prompt_includes_dialogue(self):
        prompt = make_memory_manager_training_prompt(
            dialogue_turn="Alice: I just adopted a dog named Buddy!",
            old_memory=[],
            retrieved_facts=["Alice adopted a dog named Buddy"],
        )
        assert "Buddy" in prompt
        assert "Alice" in prompt
        assert "JSON" in prompt
        assert "Structured Memory Tree" in prompt


class TestAnswerAgentPrompt:

    def test_system_prompt_contains_instructions(self):
        assert "Answer:" in ANSWER_AGENT_SYSTEM
        assert "memories" in ANSWER_AGENT_SYSTEM.lower()
        assert "timestamps" in ANSWER_AGENT_SYSTEM.lower()

    def test_system_prompt_contains_distillation_instruction(self):
        """Paper Figure 11: should instruct memory selection."""
        assert "Select memories" in ANSWER_AGENT_SYSTEM
        assert "output it before" in ANSWER_AGENT_SYSTEM

    def test_format_with_speaker_grouping(self):
        """Figure 11: memories must be grouped by speaker."""
        prompt = format_answer_agent_prompt(
            question="How many dogs does Andrew have?",
            memories_speaker_a=["Andrew adopted Buddy", "Andrew adopted Scout"],
            memories_speaker_b=["I love dogs too"],
            speaker_a_name="Andrew",
            speaker_b_name="Beth",
        )
        assert "Memories for user Andrew" in prompt
        assert "Memories for user Beth" in prompt
        assert "Buddy" in prompt
        assert "Scout" in prompt
        assert "How many dogs" in prompt
        # Should include count of memories
        assert "2 most relevant memories" in prompt

    def test_training_prompt_speaker_grouped(self):
        """Training prompt should use speaker-grouped format."""
        prompt = make_answer_agent_training_prompt(
            question="What is John's hobby?",
            memories_speaker_a=["John likes cricket"],
            memories_speaker_b=["Mary likes tennis"],
            speaker_a_name="John",
            speaker_b_name="Mary",
        )
        assert "Memories for user John" in prompt
        assert "Memories for user Mary" in prompt
        assert "hobby" in prompt
        assert "cricket" in prompt
        # Should include Memory Distillation instruction
        assert "Memories selected as relevant" in prompt
        assert "Answer:" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
