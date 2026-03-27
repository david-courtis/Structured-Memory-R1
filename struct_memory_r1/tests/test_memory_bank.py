"""Tests for memory bank CRUD operations and parsing."""
import pytest
from struct_memory_r1.memory_bank import MemoryBank, MemoryEntry, parse_memory_manager_output


class TestMemoryBank:

    def test_empty_bank(self):
        bank = MemoryBank()
        assert len(bank) == 0
        assert bank.entries == []

    def test_add(self):
        bank = MemoryBank()
        entry = bank.add("User likes pizza")
        assert entry.id == "0"
        assert entry.text == "User likes pizza"
        assert len(bank) == 1

    def test_add_multiple(self):
        bank = MemoryBank()
        bank.add("Fact 1")
        bank.add("Fact 2")
        bank.add("Fact 3")
        assert len(bank) == 3
        assert bank.get("0").text == "Fact 1"
        assert bank.get("2").text == "Fact 3"

    def test_add_with_id(self):
        bank = MemoryBank()
        entry = bank.add("Custom entry", entry_id="42")
        assert entry.id == "42"
        # Next auto-id should be 43
        entry2 = bank.add("Next entry")
        assert entry2.id == "43"

    def test_update(self):
        bank = MemoryBank()
        bank.add("User likes cheese pizza")
        success = bank.update("0", "User likes cheese and chicken pizza")
        assert success
        assert bank.get("0").text == "User likes cheese and chicken pizza"

    def test_update_nonexistent(self):
        bank = MemoryBank()
        assert bank.update("999", "text") is False

    def test_delete(self):
        bank = MemoryBank()
        bank.add("To be deleted")
        bank.add("To keep")
        assert bank.delete("0") is True
        assert len(bank) == 1
        assert bank.get("0") is None
        assert bank.get("1").text == "To keep"

    def test_delete_nonexistent(self):
        bank = MemoryBank()
        assert bank.delete("999") is False

    def test_copy(self):
        bank = MemoryBank()
        bank.add("Entry 1")
        bank.add("Entry 2")
        copy = bank.copy()
        copy.update("0", "Modified")
        # Original should be unchanged
        assert bank.get("0").text == "Entry 1"
        assert copy.get("0").text == "Modified"

    def test_from_list(self):
        entries = [
            {"id": "0", "text": "User is a software engineer"},
            {"id": "1", "text": "User likes cricket"},
        ]
        bank = MemoryBank.from_list(entries)
        assert len(bank) == 2
        assert bank.get("0").text == "User is a software engineer"

    def test_to_list(self):
        bank = MemoryBank()
        bank.add("Fact A")
        bank.add("Fact B")
        lst = bank.to_list()
        assert lst == [
            {"id": "0", "text": "Fact A"},
            {"id": "1", "text": "Fact B"},
        ]

    def test_structured_insert_interaction(self):
        bank = MemoryBank()
        entries = bank.insert_interaction(
            interaction="Alice: I adopted a dog named Buddy from the shelter.",
            timestamp="2023-01-01 10:00",
        )
        assert len(entries) == 1
        structured = bank.to_structured_list()[0]
        assert structured["speaker"] == "Alice"
        assert structured["path"][0] == "Alice"
        assert structured["timestamp"] == "2023-01-01 10:00"

    def test_structured_retrieve(self):
        bank = MemoryBank()
        bank.insert_interaction("Alice: I adopted a dog named Buddy.")
        bank.insert_interaction("Bob: I bought a new guitar.")
        results = bank.retrieve("What is the dog's name?", topk=1, speaker="Alice")
        assert len(results) == 1
        assert "Buddy" in results[0]["text"]

    def test_from_structured_list(self):
        bank = MemoryBank.from_list([
            {
                "id": "0",
                "text": "Alice adopted a dog named Buddy",
                "speaker": "Alice",
                "topic": "dog",
                "path": ["Alice", "dog"],
                "timestamp": "2023-01-01",
            }
        ])
        retrieved = bank.retrieve("dog buddy", topk=1, speaker="Alice")
        assert len(retrieved) == 1
        assert retrieved[0]["path"] == ["Alice", "dog"]

    def test_global_query_annotation_planner(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Buddy is a golden retriever", "speaker": "Alice", "topic": "pets", "path": ["Alice", "pets", "dogs", "Buddy", "profile"]},
            {"id": "1", "text": "Scout needs confidence work", "speaker": "Alice", "topic": "pets", "path": ["Alice", "pets", "dogs", "Scout", "training"]},
            {"id": "2", "text": "Bob works at the hospital", "speaker": "Bob", "topic": "work", "path": ["Bob", "work", "hospital"]},
        ])
        planner_calls = []

        def planner(**kwargs):
            planner_calls.append(kwargs)
            return {
                "full_query": kwargs["base_query"],
                "speaker": "Alice",
                "topic": "pets",
                "subtopic": "dogs",
                "entity": "Buddy",
                "time_hint": "none",
            }

        retrieved = bank.retrieve("What breed is Buddy?", topk=1, planner=planner)
        assert len(planner_calls) == 1
        assert planner_calls[0]["schema"]["speakers"] == ["Alice", "Bob"]
        assert retrieved[0]["id"] == "0"
        assert "Buddy is a golden retriever" in retrieved[0]["text"]

    def test_apply_operations_with_structured_metadata(self):
        bank = MemoryBank()
        ops = [{
            "id": "0",
            "text": "Alice adopted a dog named Buddy",
            "event": "ADD",
            "speaker": "Alice",
            "topic": "dog",
            "path": ["Alice", "dog"],
        }]
        new_bank = bank.apply_operations(ops)
        structured = new_bank.to_structured_list()[0]
        assert structured["speaker"] == "Alice"
        assert structured["topic"] == "dog"
        assert structured["path"] == ["Alice", "dog"]

    def test_create_subtopic_operation(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Alice adopted Buddy", "path": ["Alice", "pets"]},
        ])
        new_bank = bank.apply_operations([
            {
                "event": "CREATE_SUBTOPIC",
                "parent_path": ["Alice", "pets"],
                "path": ["Alice", "pets", "dogs"],
                "text": "dogs",
            }
        ])
        tree = new_bank.to_tree_dict()
        alice_node = tree["children"][0]
        pets_node = alice_node["children"][0]
        assert pets_node["key"] == "pets"
        assert pets_node["children"][0]["key"] == "dogs"

    def test_move_entry_operation(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Alice adopted Buddy", "path": ["Alice", "pets"]},
        ])
        new_bank = bank.apply_operations([
            {
                "id": "0",
                "event": "MOVE",
                "path": ["Alice", "pets", "dogs"],
            }
        ])
        assert new_bank.get("0").path == ["Alice", "pets", "dogs"]

    def test_move_entry_rejects_bad_merge_without_force(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Buddy is a golden retriever", "path": ["Alice", "golden"]},
        ])
        new_bank = bank.apply_operations([
            {
                "id": "0",
                "event": "MOVE",
                "path": ["Alice", "adopted", "Scout"],
            }
        ])
        assert new_bank.get("0").path == ["Alice", "golden"]

    def test_move_entry_allows_force_override(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Buddy is a golden retriever", "path": ["Alice", "golden"]},
        ])
        new_bank = bank.apply_operations([
            {
                "id": "0",
                "event": "MOVE",
                "path": ["Alice", "adopted", "Scout"],
                "force": True,
            }
        ])
        assert new_bank.get("0").path == ["Alice", "adopted", "Scout"]

    def test_move_subtree_operation(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Alice adopted Buddy", "path": ["Alice", "pets", "dogs"]},
            {"id": "1", "text": "Alice adopted Scout", "path": ["Alice", "pets", "dogs"]},
        ])
        new_bank = bank.apply_operations([
            {
                "event": "MOVE_NODE",
                "source_path": ["Alice", "pets", "dogs"],
                "path": ["Alice", "animals", "dogs"],
            }
        ])
        assert new_bank.get("0").path == ["Alice", "animals", "dogs"]
        assert new_bank.get("1").path == ["Alice", "animals", "dogs"]

    def test_split_topic_operation(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Alice adopted Buddy the dog", "path": ["Alice", "pets"]},
            {"id": "1", "text": "Alice adopted Whiskers the cat", "path": ["Alice", "pets"]},
        ])
        new_bank = bank.apply_operations([
            {
                "event": "SPLIT_TOPIC",
                "path": ["Alice", "pets"],
                "subtopics": ["dog", "cat"],
            }
        ])
        assert new_bank.get("0").path == ["Alice", "pets", "dog"]
        assert new_bank.get("1").path == ["Alice", "pets", "cat"]

    def test_merge_topic_operation(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Alice likes basketball", "path": ["Alice", "sports", "playing"]},
            {"id": "1", "text": "Alice watches soccer", "path": ["Alice", "sports", "watching"]},
        ])
        new_bank = bank.apply_operations([
            {
                "event": "MERGE_TOPIC",
                "source_paths": [
                    ["Alice", "sports", "playing"],
                    ["Alice", "sports", "watching"],
                ],
                "path": ["Alice", "sports", "ball"],
            }
        ])
        assert new_bank.get("0").path == ["Alice", "sports", "ball"]
        assert new_bank.get("1").path == ["Alice", "sports", "ball"]

    def test_apply_operations_add(self):
        bank = MemoryBank()
        bank.add("Existing entry")
        ops = [{"id": "1", "text": "New entry", "event": "ADD"}]
        new_bank = bank.apply_operations(ops)
        assert len(new_bank) == 2
        assert len(bank) == 1  # Original unchanged

    def test_apply_operations_update(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "I really like cheese pizza"},
            {"id": "1", "text": "User likes to play cricket"},
        ])
        ops = [
            {"id": "0", "text": "Loves cheese and chicken pizza", "event": "UPDATE",
             "old_memory": "I really like cheese pizza"},
            {"id": "1", "text": "Loves to play cricket with friends", "event": "UPDATE",
             "old_memory": "User likes to play cricket"},
        ]
        new_bank = bank.apply_operations(ops)
        assert new_bank.get("0").text == "Loves cheese and chicken pizza"
        assert new_bank.get("1").text == "Loves to play cricket with friends"

    def test_apply_operations_delete(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Loves cheese pizza"},
        ])
        ops = [{"id": "0", "text": "Loves cheese pizza", "event": "DELETE"}]
        new_bank = bank.apply_operations(ops)
        assert len(new_bank) == 0

    def test_apply_operations_noop(self):
        bank = MemoryBank.from_list([
            {"id": "0", "text": "Name is John"},
        ])
        ops = [{"id": "0", "text": "Name is John", "event": "NONE"}]
        new_bank = bank.apply_operations(ops)
        assert new_bank.get("0").text == "Name is John"

    def test_apply_operations_mixed(self):
        """Test the example from paper Figure 9: ADD + UPDATE scenario."""
        bank = MemoryBank.from_list([
            {"id": "0", "text": "I really like cheese pizza"},
            {"id": "2", "text": "User likes to play cricket"},
        ])
        ops = [
            {"id": "0", "text": "Loves cheese and chicken pizza",
             "event": "UPDATE", "old_memory": "I really like cheese pizza"},
            {"id": "2", "text": "Loves to play cricket with friends",
             "event": "UPDATE", "old_memory": "User likes to play cricket"},
        ]
        new_bank = bank.apply_operations(ops)
        assert len(new_bank) == 2
        assert new_bank.get("0").text == "Loves cheese and chicken pizza"

    def test_paper_example_dog_adoption(self):
        """
        Test the key example from paper Figure 1:
        User first says they adopted a dog named Buddy, then later says
        they adopted another dog named Scout. The RL-trained manager should
        UPDATE (not DELETE+ADD), consolidating into one entry.
        """
        bank = MemoryBank()
        bank.add("Andrew adopted a new dog from a shelter and named him Buddy.")

        # The RL-trained manager issues UPDATE (not DELETE+ADD)
        ops = [{
            "id": "0",
            "text": "Andrew adopted a dog from a shelter and named him Buddy, "
                    "and later adopted another dog named Scout.",
            "event": "UPDATE",
            "old_memory": "Andrew adopted a new dog from a shelter and named him Buddy."
        }]
        new_bank = bank.apply_operations(ops)
        assert len(new_bank) == 1
        assert "Buddy" in new_bank.get("0").text
        assert "Scout" in new_bank.get("0").text


class TestParseMemoryManagerOutput:

    def test_valid_json(self):
        output = '''
{
    "memory": [
        {"id": "0", "text": "User is a software engineer", "event": "NONE"},
        {"id": "1", "text": "Name is John", "event": "ADD"}
    ]
}'''
        ops, success = parse_memory_manager_output(output)
        assert success
        assert len(ops) == 2
        assert ops[1]["event"] == "ADD"

    def test_json_with_surrounding_text(self):
        output = '''Here is my analysis:
{
    "memory": [
        {"id": "0", "text": "Fact", "event": "NONE"}
    ]
}
That's my decision.'''
        ops, success = parse_memory_manager_output(output)
        assert success
        assert len(ops) == 1

    def test_invalid_json(self):
        output = "I don't know what to do with this memory."
        ops, success = parse_memory_manager_output(output)
        assert not success
        assert ops == []

    def test_empty_memory_list(self):
        output = '{"memory": []}'
        ops, success = parse_memory_manager_output(output)
        assert success
        assert ops == []

    def test_json_in_code_block(self):
        output = '''```json
{
    "memory": [
        {"id": "0", "text": "Test", "event": "ADD"}
    ]
}
```'''
        ops, success = parse_memory_manager_output(output)
        assert success

    def test_structured_memory_key(self):
        output = '''
{
    "structured_memory": [
        {"id": "0", "text": "Alice adopted Buddy", "event": "ADD", "path": ["Alice", "dog"]}
    ]
}'''
        ops, success = parse_memory_manager_output(output)
        assert success
        assert ops[0]["path"] == ["Alice", "dog"]

    def test_parse_structure_operation(self):
        output = '''
{
    "memory": [
        {"event": "SPLIT_TOPIC", "path": ["Alice", "pets"], "subtopics": ["dog", "cat"]}
    ]
}'''
        ops, success = parse_memory_manager_output(output)
        assert success
        assert ops[0]["event"] == "SPLIT_TOPIC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
