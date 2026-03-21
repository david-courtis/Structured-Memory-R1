"""Tests for memory bank CRUD operations and parsing."""
import pytest
from memory_r1.memory_bank import MemoryBank, MemoryEntry, parse_memory_manager_output


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
