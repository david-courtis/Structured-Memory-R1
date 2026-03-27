"""
Structured memory bank with CRUD operations for Memory-R1 style training.

The original Memory-R1 implementation used a flat list of ``{id, text}``
memories. This version keeps that flat interface for compatibility, while also
maintaining a lightweight tree structure for insertion and retrieval:

root -> speaker -> topic -> fact node

The Memory Manager can still emit the original JSON schema, but may optionally
include structured metadata such as ``speaker``, ``topic`` and ``path``.
"""
import json
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


class MemoryEntry:
    """A single memory entry with flat text and optional structured metadata."""

    def __init__(
        self,
        id: str,
        text: str,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        timestamp: Optional[str] = None,
        path: Optional[List[str]] = None,
        node_type: str = "fact",
    ):
        self.id = id
        self.text = text
        self.speaker = speaker
        self.topic = topic
        self.timestamp = timestamp
        self.path = list(path or [])
        self.node_type = node_type

    def to_dict(self, include_structure: bool = False) -> dict:
        data = {"id": self.id, "text": self.text}
        if include_structure:
            if self.speaker:
                data["speaker"] = self.speaker
            if self.topic:
                data["topic"] = self.topic
            if self.timestamp:
                data["timestamp"] = self.timestamp
            if self.path:
                data["path"] = list(self.path)
            if self.node_type:
                data["node_type"] = self.node_type
        return data

    def __repr__(self):
        return (
            "MemoryEntry("
            f"id={self.id!r}, text={self.text!r}, speaker={self.speaker!r}, "
            f"topic={self.topic!r}, path={self.path!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, MemoryEntry):
            return False
        return (
            self.id == other.id
            and self.text == other.text
            and self.speaker == other.speaker
            and self.topic == other.topic
            and self.timestamp == other.timestamp
            and self.path == other.path
            and self.node_type == other.node_type
        )


class StructuredMemoryNode:
    """A node in the structured memory tree."""

    def __init__(
        self,
        key: str,
        node_type: str,
        parent: Optional["StructuredMemoryNode"] = None,
    ):
        self.key = key
        self.node_type = node_type
        self.parent = parent
        self.children: Dict[str, "StructuredMemoryNode"] = {}
        self.entry_ids: Set[str] = set()

    @property
    def path(self) -> List[str]:
        node: Optional["StructuredMemoryNode"] = self
        parts: List[str] = []
        while node is not None and node.parent is not None:
            parts.append(node.key)
            node = node.parent
        return list(reversed(parts))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "node_type": self.node_type,
            "entries": sorted(self.entry_ids),
            "children": [child.to_dict() for child in self.children.values()],
        }


class MemoryBank:
    """
    Compatibility wrapper around a structured tree-backed memory bank.

    The bank preserves the flat ``{id, text}`` interface used throughout the
    original Memory-R1 code while additionally supporting:
    - raw interaction insertion into a speaker/topic tree
    - structured retrieval over tree nodes
    - optional structured metadata in manager operations
    """

    def __init__(self):
        self._entries: Dict[str, MemoryEntry] = {}
        self._next_id: int = 0
        self._root = StructuredMemoryNode(key="root", node_type="root", parent=None)

    @property
    def entries(self) -> List[MemoryEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    @staticmethod
    def _normalize_path(path: Optional[Iterable[str]]) -> List[str]:
        if path is None:
            return []
        return [str(part).strip() for part in path if str(part).strip()]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    @classmethod
    def _infer_topic(cls, text: str) -> str:
        tokens = cls._tokenize(text)
        stopwords = {
            "the", "and", "with", "from", "that", "this", "have", "just", "about",
            "your", "they", "them", "then", "into", "been", "were", "what", "when",
            "where", "which", "would", "could", "should", "there", "their", "because",
            "really", "also", "after", "before", "while", "will", "said", "says",
        }
        candidates = [tok for tok in tokens if len(tok) > 3 and tok not in stopwords]
        if not candidates:
            return "general"
        return candidates[0]

    def _resolve_path(
        self,
        text: str,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        path: Optional[Iterable[str]] = None,
    ) -> List[str]:
        normalized = self._normalize_path(path)
        if normalized:
            return normalized

        speaker_key = (speaker or "global").strip() or "global"
        topic_key = (topic or self._infer_topic(text)).strip() or "general"
        return [speaker_key, topic_key]

    def _ensure_path(self, path: List[str]) -> StructuredMemoryNode:
        node = self._root
        for depth, part in enumerate(path):
            node_type = "speaker" if depth == 0 else "topic"
            if part not in node.children:
                node.children[part] = StructuredMemoryNode(
                    key=part,
                    node_type=node_type,
                    parent=node,
                )
            node = node.children[part]
        return node

    def _detach_entry_from_tree(self, entry: MemoryEntry):
        if not entry.path:
            return

        node = self._root
        visited = [self._root]
        for part in entry.path:
            node = node.children.get(part)
            if node is None:
                return
            visited.append(node)

        node.entry_ids.discard(entry.id)

        for current in reversed(visited[1:]):
            parent = current.parent
            if parent is None:
                continue
            if current.entry_ids or current.children:
                continue
            parent.children.pop(current.key, None)

    def add(
        self,
        text: str,
        entry_id: Optional[str] = None,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        timestamp: Optional[str] = None,
        path: Optional[Iterable[str]] = None,
        node_type: str = "fact",
    ) -> MemoryEntry:
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
        resolved_path = self._resolve_path(text=text, speaker=speaker, topic=topic, path=path)
        topic_value = topic or (resolved_path[1] if len(resolved_path) > 1 else self._infer_topic(text))
        speaker_value = speaker or (resolved_path[0] if resolved_path else None)
        entry = MemoryEntry(
            id=entry_id,
            text=text,
            speaker=speaker_value,
            topic=topic_value,
            timestamp=timestamp,
            path=resolved_path,
            node_type=node_type,
        )
        self._entries[entry_id] = entry
        self._ensure_path(resolved_path).entry_ids.add(entry_id)
        return entry

    def update(
        self,
        entry_id: str,
        new_text: str,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        timestamp: Optional[str] = None,
        path: Optional[Iterable[str]] = None,
    ) -> bool:
        """Update an existing entry's text. Returns False if ID not found."""
        if entry_id not in self._entries:
            return False
        entry = self._entries[entry_id]
        self._detach_entry_from_tree(entry)
        resolved_path = self._resolve_path(
            text=new_text,
            speaker=speaker or entry.speaker,
            topic=topic or entry.topic,
            path=path or entry.path,
        )
        entry.text = new_text
        entry.speaker = speaker or entry.speaker or (resolved_path[0] if resolved_path else None)
        entry.topic = topic or entry.topic or (resolved_path[1] if len(resolved_path) > 1 else None)
        entry.timestamp = timestamp or entry.timestamp
        entry.path = resolved_path
        self._ensure_path(resolved_path).entry_ids.add(entry_id)
        return True

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns False if ID not found."""
        if entry_id not in self._entries:
            return False
        self._detach_entry_from_tree(self._entries[entry_id])
        del self._entries[entry_id]
        return True

    def to_list(self, include_structure: bool = False) -> List[dict]:
        """Serialize to list of {id, text} dicts (for prompt injection)."""
        return [e.to_dict(include_structure=include_structure) for e in self._entries.values()]

    def to_structured_list(self) -> List[dict]:
        """Serialize including structured metadata."""
        return self.to_list(include_structure=True)

    def to_json(self) -> str:
        return json.dumps(self.to_list(), indent=2)

    def copy(self) -> "MemoryBank":
        """Deep copy the memory bank."""
        new_bank = MemoryBank()
        for entry in self._entries.values():
            new_bank.add(
                text=entry.text,
                entry_id=entry.id,
                speaker=entry.speaker,
                topic=entry.topic,
                timestamp=entry.timestamp,
                path=entry.path,
                node_type=entry.node_type,
            )
        new_bank._next_id = self._next_id
        return new_bank

    @classmethod
    def from_list(cls, entries: List[dict]) -> "MemoryBank":
        """Create a MemoryBank from flat or structured entry dicts."""
        bank = cls()
        for e in entries:
            bank.add(
                text=e["text"],
                entry_id=str(e["id"]),
                speaker=e.get("speaker"),
                topic=e.get("topic"),
                timestamp=e.get("timestamp"),
                path=e.get("path"),
                node_type=e.get("node_type", "fact"),
            )
        return bank

    def insert_interaction(
        self,
        interaction: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None,
        entry_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Turn a raw interaction into one or more tree leaves and insert them.

        A simple sentence splitter is sufficient here. Each sentence becomes
        one fact leaf under ``speaker/topic``.
        """
        text = (interaction or "").strip()
        if not text:
            return []

        interaction_speaker = speaker
        raw_text = text
        if ":" in text and speaker is None:
            candidate_speaker, remainder = text.split(":", 1)
            if candidate_speaker.strip() and remainder.strip():
                interaction_speaker = candidate_speaker.strip()
                raw_text = remainder.strip()
        elif speaker is not None and text.startswith(f"{speaker}:"):
            raw_text = text[len(f"{speaker}:"):].strip()

        sentences = [segment.strip() for segment in re.split(r"[.!?]+", raw_text) if segment.strip()]
        if not sentences:
            sentences = [raw_text]

        inserted: List[MemoryEntry] = []
        current_id = entry_id
        for sentence in sentences:
            inserted.append(
                self.add(
                    text=f"{interaction_speaker}: {sentence}" if interaction_speaker else sentence,
                    entry_id=current_id,
                    speaker=interaction_speaker,
                    topic=self._infer_topic(sentence),
                    timestamp=timestamp,
                )
            )
            current_id = None
        return inserted

    def retrieve(
        self,
        query: str,
        topk: int = 5,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve relevant memories from the tree with simple token overlap."""
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            query_tokens = set(self._tokenize(topic or "general"))

        results: List[Tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            if speaker and (entry.speaker or "").lower() != speaker.lower():
                continue
            if topic and (entry.topic or "").lower() != topic.lower():
                continue

            entry_tokens = set(self._tokenize(entry.text))
            if entry.speaker:
                entry_tokens.update(self._tokenize(entry.speaker))
            if entry.topic:
                entry_tokens.update(self._tokenize(entry.topic))
            for part in entry.path:
                entry_tokens.update(self._tokenize(part))

            overlap = len(query_tokens & entry_tokens)
            if overlap == 0 and query:
                continue

            score = overlap / max(1, len(query_tokens))
            if speaker and entry.speaker and entry.speaker.lower() == speaker.lower():
                score += 0.1
            if topic and entry.topic and entry.topic.lower() == topic.lower():
                score += 0.1
            results.append((score, entry))

        results.sort(key=lambda item: (-item[0], item[1].id))
        return [
            {
                "id": entry.id,
                "text": entry.text,
                "speaker": entry.speaker,
                "topic": entry.topic,
                "timestamp": entry.timestamp,
                "path": list(entry.path),
                "score": float(score),
            }
            for score, entry in results[:topk]
        ]

    def to_tree_dict(self) -> dict:
        """Serialize the tree for prompts or debugging."""
        return self._root.to_dict()

    def to_prompt_payload(self) -> List[dict]:
        """Structured prompt view that remains JSON serializable."""
        return self.to_structured_list()

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
            speaker = op.get("speaker")
            topic = op.get("topic")
            timestamp = op.get("timestamp")
            path = op.get("path")

            if event == "ADD":
                new_bank.add(
                    text=text,
                    entry_id=entry_id,
                    speaker=speaker,
                    topic=topic,
                    timestamp=timestamp,
                    path=path,
                )
            elif event == "UPDATE":
                if not new_bank.update(
                    entry_id,
                    text,
                    speaker=speaker,
                    topic=topic,
                    timestamp=timestamp,
                    path=path,
                ):
                    # If ID doesn't exist, add it instead
                    new_bank.add(
                        text=text,
                        entry_id=entry_id,
                        speaker=speaker,
                        topic=topic,
                        timestamp=timestamp,
                        path=path,
                    )
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
    json_pattern = r'\{[\s\S]*"(?:memory|structured_memory)"[\s\S]*\}'
    matches = list(re.finditer(json_pattern, output_text))

    if not matches:
        return [], False

    # Try each match (prefer the last one, as in the answer extraction logic)
    for match in reversed(matches):
        try:
            parsed = json.loads(match.group())
            memory_items = parsed.get("memory")
            if isinstance(memory_items, list):
                return memory_items, True
            structured_items = parsed.get("structured_memory")
            if isinstance(structured_items, list):
                return structured_items, True
        except json.JSONDecodeError:
            continue

    return [], False
