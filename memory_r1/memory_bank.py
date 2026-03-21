"""
Flat memory bank with CRUD operations for Memory-R1.

Following Memory-R1 (Yan et al., 2025), the memory bank is a list of entries
with {id, text} pairs supporting ADD, UPDATE, DELETE, NOOP operations.
The Memory Manager outputs structured JSON specifying operations and content.
"""
import json
import copy
import re
from typing import List, Dict, Optional, Tuple


class MemoryEntry:
    """A single memory entry with id and text content."""

    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text}

    def __repr__(self):
        return f"MemoryEntry(id={self.id!r}, text={self.text!r})"

    def __eq__(self, other):
        if not isinstance(other, MemoryEntry):
            return False
        return self.id == other.id and self.text == other.text


class MemoryBank:
    """
    Flat memory bank supporting ADD, UPDATE, DELETE, NOOP operations.

    Following the Memory-R1 paper (Figures 9-10), the memory bank is a list
    of {id, text} entries. Operations are:
    - ADD: Insert a new entry with a new ID
    - UPDATE: Modify an existing entry's text, preserving its ID
    - DELETE: Remove an entry by ID
    - NOOP (NONE): No change
    """

    def __init__(self):
        self._entries: Dict[str, MemoryEntry] = {}
        self._next_id: int = 0

    @property
    def entries(self) -> List[MemoryEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    def add(self, text: str, entry_id: Optional[str] = None) -> MemoryEntry:
        """Add a new entry. Auto-assigns ID if not provided."""
        if entry_id is None:
            entry_id = str(self._next_id)
            self._next_id += 1
        else:
            # Keep _next_id ahead of any manually assigned IDs
            try:
                numeric_id = int(entry_id)
                if numeric_id >= self._next_id:
                    self._next_id = numeric_id + 1
            except ValueError:
                pass
        entry = MemoryEntry(id=entry_id, text=text)
        self._entries[entry_id] = entry
        return entry

    def update(self, entry_id: str, new_text: str) -> bool:
        """Update an existing entry's text. Returns False if ID not found."""
        if entry_id not in self._entries:
            return False
        self._entries[entry_id].text = new_text
        return True

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns False if ID not found."""
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        return True

    def to_list(self) -> List[dict]:
        """Serialize to list of {id, text} dicts (for prompt injection)."""
        return [e.to_dict() for e in self._entries.values()]

    def to_json(self) -> str:
        return json.dumps(self.to_list(), indent=2)

    def copy(self) -> "MemoryBank":
        """Deep copy the memory bank."""
        new_bank = MemoryBank()
        new_bank._entries = {k: MemoryEntry(id=v.id, text=v.text)
                             for k, v in self._entries.items()}
        new_bank._next_id = self._next_id
        return new_bank

    @classmethod
    def from_list(cls, entries: List[dict]) -> "MemoryBank":
        """Create a MemoryBank from a list of {id, text} dicts."""
        bank = cls()
        for e in entries:
            bank.add(text=e["text"], entry_id=str(e["id"]))
        return bank

    def apply_operations(self, operations: List[dict]) -> "MemoryBank":
        """
        Apply a list of operations from Memory Manager output.

        Each operation dict has:
        - "id": str - the entry ID
        - "text": str - the entry text
        - "event": str - one of "ADD", "UPDATE", "DELETE", "NONE"/"NOOP"
        - "old_memory": str (optional) - for UPDATE, the old text

        Returns a new MemoryBank with operations applied.
        """
        new_bank = self.copy()
        for op in operations:
            event = op.get("event", "NONE").upper()
            entry_id = str(op.get("id", ""))
            text = op.get("text", "")

            if event == "ADD":
                new_bank.add(text=text, entry_id=entry_id)
            elif event == "UPDATE":
                if not new_bank.update(entry_id, text):
                    # If ID doesn't exist, add it instead
                    new_bank.add(text=text, entry_id=entry_id)
            elif event == "DELETE":
                new_bank.delete(entry_id)
            elif event in ("NONE", "NOOP"):
                pass  # No change
        return new_bank


def parse_memory_manager_output(output_text: str) -> Tuple[List[dict], bool]:
    """
    Parse the Memory Manager's JSON output into a list of operations.

    The Memory Manager outputs JSON like (from paper Figure 9):
    {
        "memory": [
            {"id": "0", "text": "...", "event": "NONE"},
            {"id": "1", "text": "...", "event": "UPDATE", "old_memory": "..."},
            {"id": "2", "text": "...", "event": "ADD"}
        ]
    }

    Returns:
        (operations, success): list of operation dicts and whether parsing succeeded
    """
    # Try to extract JSON from the output
    # The model may wrap it in markdown code blocks or have extra text
    json_pattern = r'\{[\s\S]*"memory"[\s\S]*\}'
    matches = list(re.finditer(json_pattern, output_text))

    if not matches:
        return [], False

    # Try each match (prefer the last one, as in the answer extraction logic)
    for match in reversed(matches):
        try:
            parsed = json.loads(match.group())
            if "memory" in parsed and isinstance(parsed["memory"], list):
                return parsed["memory"], True
        except json.JSONDecodeError:
            continue

    return [], False
