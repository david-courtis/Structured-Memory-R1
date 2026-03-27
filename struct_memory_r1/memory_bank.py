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
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


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
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_model_name = None
        self._embedding_model = None
        self._embedding_tokenizer = None

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

    @staticmethod
    def _entry_retrieval_text(entry: MemoryEntry) -> str:
        """Build a similarity-search document for a memory entry."""
        parts: List[str] = []
        if entry.speaker:
            parts.append(f"speaker {entry.speaker}")
        if entry.topic:
            parts.append(f"topic {entry.topic}")
        if entry.path:
            parts.append("path " + " ".join(entry.path))
        parts.append(entry.text)
        return " ".join(part for part in parts if part).strip()

    def _get_local_embedding_backend(self, model: str):
        if self._embedding_model is not None and self._embedding_model_name == model:
            return self._embedding_model

        if SentenceTransformer is not None:
            try:
                backend = SentenceTransformer(model, trust_remote_code=True)
                self._embedding_model_name = model
                self._embedding_model = backend
                self._embedding_tokenizer = None
                return backend
            except Exception:
                pass

        if AutoTokenizer is None or AutoModel is None or torch is None:
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            hf_model = AutoModel.from_pretrained(model, trust_remote_code=True)
            hf_model.eval()
            self._embedding_model_name = model
            self._embedding_model = hf_model
            self._embedding_tokenizer = tokenizer
            return hf_model
        except Exception:
            return None

    def _mean_pool_hf_embedding(self, text: str) -> Optional[List[float]]:
        if self._embedding_model is None or self._embedding_tokenizer is None or torch is None:
            return None
        encoded = self._embedding_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with torch.no_grad():
            outputs = self._embedding_model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized[0].cpu().tolist()

    def _get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        normalized = text.strip()
        if not normalized:
            return None
        cache_key = f"{model}:{normalized}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        backend = self._get_local_embedding_backend(model)
        if backend is None:
            return None
        try:
            if hasattr(backend, "encode"):
                encoded = backend.encode(
                    [normalized],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                embedding = encoded[0].tolist() if hasattr(encoded[0], "tolist") else list(encoded[0])
            elif SentenceTransformer is not None and isinstance(backend, SentenceTransformer):
                embedding = backend.encode(
                    [normalized],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0].tolist()
            else:
                embedding = self._mean_pool_hf_embedding(normalized)
                if embedding is None:
                    return None
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        numerator = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            numerator += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return numerator / ((norm_a ** 0.5) * (norm_b ** 0.5))

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

    def _get_node(self, path: Optional[Iterable[str]]) -> Optional[StructuredMemoryNode]:
        normalized = self._normalize_path(path)
        node = self._root
        for part in normalized:
            node = node.children.get(part)
            if node is None:
                return None
        return node

    def _collect_entry_ids_under_node(self, node: Optional[StructuredMemoryNode]) -> List[str]:
        if node is None:
            return []
        collected: List[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            collected.extend(sorted(current.entry_ids))
            stack.extend(current.children.values())
        return collected

    def _remove_empty_ancestors(self, node: Optional[StructuredMemoryNode]):
        current = node
        while current is not None and current.parent is not None:
            parent = current.parent
            if current.entry_ids or current.children:
                break
            parent.children.pop(current.key, None)
            current = parent

    def _drop_subtree(self, path: Optional[Iterable[str]]):
        node = self._get_node(path)
        if node is None or node.parent is None:
            return
        parent = node.parent
        parent.children.pop(node.key, None)
        self._remove_empty_ancestors(parent)

    def _topic_from_path(self, path: List[str], fallback_text: str = "") -> Optional[str]:
        if len(path) >= 2:
            return path[-1]
        if len(path) == 1:
            return path[0]
        if fallback_text:
            return self._infer_topic(fallback_text)
        return None

    def _speaker_from_path(self, path: List[str], fallback: Optional[str] = None) -> Optional[str]:
        if path:
            return path[0]
        return fallback

    def _move_entry_to_path(self, entry_id: str, new_path: Iterable[str]):
        entry = self.get(entry_id)
        if entry is None:
            return
        normalized_path = self._normalize_path(new_path)
        self._detach_entry_from_tree(entry)
        entry.path = normalized_path
        entry.speaker = self._speaker_from_path(normalized_path, fallback=entry.speaker)
        entry.topic = self._topic_from_path(normalized_path, fallback_text=entry.text) or entry.topic
        self._ensure_path(normalized_path).entry_ids.add(entry_id)

    def _move_subtree(self, source_path: Iterable[str], target_path: Iterable[str]):
        source = self._normalize_path(source_path)
        target = self._normalize_path(target_path)
        if not source or source == target:
            return
        node = self._get_node(source)
        if node is None:
            return
        entry_ids = self._collect_entry_ids_under_node(node)
        for entry_id in entry_ids:
            entry = self.get(entry_id)
            if entry is None:
                continue
            suffix = entry.path[len(source):] if entry.path[:len(source)] == source else []
            self._move_entry_to_path(entry_id, target + suffix)
        self._drop_subtree(source)

    def _score_subtopic_fit(self, entry: MemoryEntry, subtopic: str) -> float:
        subtopic_tokens = set(self._tokenize(subtopic))
        if not subtopic_tokens:
            return 0.0
        entry_tokens = set(self._tokenize(self._entry_retrieval_text(entry)))
        overlap = len(subtopic_tokens & entry_tokens)
        if overlap == 0:
            inferred = self._infer_topic(entry.text)
            if inferred == subtopic.lower():
                return 0.5
        return overlap / max(1, len(subtopic_tokens))

    def _assign_entries_to_subtopics(
        self,
        entry_ids: List[str],
        subtopics: List[str],
        assignments: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        bucketed = {subtopic: [] for subtopic in subtopics}
        remaining_ids = list(entry_ids)
        if assignments:
            remaining_ids = []
            assigned = set()
            for subtopic, ids in assignments.items():
                if subtopic not in bucketed:
                    continue
                for entry_id in ids:
                    if entry_id in self._entries and entry_id not in assigned:
                        bucketed[subtopic].append(entry_id)
                        assigned.add(entry_id)
            remaining_ids = [entry_id for entry_id in entry_ids if entry_id not in assigned]

        for entry_id in remaining_ids:
            entry = self.get(entry_id)
            if entry is None:
                continue
            scored = sorted(
                ((self._score_subtopic_fit(entry, subtopic), subtopic) for subtopic in subtopics),
                key=lambda item: (-item[0], item[1]),
            )
            chosen = scored[0][1] if scored else subtopics[0]
            bucketed[chosen].append(entry_id)
        return bucketed

    def _create_subtopic(self, path: Iterable[str], parent_path: Optional[Iterable[str]] = None):
        normalized = self._normalize_path(path)
        if not normalized and parent_path is not None:
            normalized = self._normalize_path(parent_path)
        if normalized:
            self._ensure_path(normalized)

    def _split_topic(
        self,
        path: Iterable[str],
        subtopics: List[str],
        assignments: Optional[Dict[str, List[str]]] = None,
    ):
        normalized = self._normalize_path(path)
        if not normalized or not subtopics:
            return
        node = self._get_node(normalized)
        if node is None:
            self._ensure_path(normalized)
            node = self._get_node(normalized)
        if node is None:
            return

        direct_entry_ids = sorted(node.entry_ids)
        node.entry_ids.clear()
        bucketed = self._assign_entries_to_subtopics(direct_entry_ids, subtopics, assignments)
        for subtopic in subtopics:
            child_path = normalized + [subtopic]
            self._ensure_path(child_path)
            for entry_id in bucketed.get(subtopic, []):
                self._move_entry_to_path(entry_id, child_path)
        self._remove_empty_ancestors(node)

    def _merge_topics(
        self,
        source_paths: List[Iterable[str]],
        target_path: Iterable[str],
    ):
        normalized_target = self._normalize_path(target_path)
        if not normalized_target:
            return
        self._ensure_path(normalized_target)
        for source_path in source_paths:
            normalized_source = self._normalize_path(source_path)
            if not normalized_source or normalized_source == normalized_target:
                continue
            node = self._get_node(normalized_source)
            if node is None:
                continue
            entry_ids = self._collect_entry_ids_under_node(node)
            for entry_id in entry_ids:
                entry = self.get(entry_id)
                if entry is None:
                    continue
                suffix = entry.path[len(normalized_source):] if entry.path[:len(normalized_source)] == normalized_source else []
                self._move_entry_to_path(entry_id, normalized_target + suffix)
            self._drop_subtree(normalized_source)

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
        """
        Retrieve relevant memories with similarity-based ranking.

        Speaker/topic information is included as retrieval tokens instead of
        acting as a strict tree filter. Exact speaker/topic matches receive a
        small bonus after similarity scoring.
        """
        if not self._entries:
            return []

        query_parts = [query.strip()] if query else []
        if speaker:
            query_parts.append(f"speaker {speaker}")
        if topic:
            query_parts.append(f"topic {topic}")
        retrieval_query = " ".join(part for part in query_parts if part).strip() or "general"

        results: List[Tuple[float, MemoryEntry]] = []
        entries = list(self._entries.values())
        backend = os.environ.get("STRUCT_MEMORY_R1_RETRIEVAL_BACKEND", "auto").lower()
        embedding_model = os.environ.get("STRUCT_MEMORY_R1_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

        use_vector_backend = backend in {"auto", "local", "huggingface", "vector", "embedding"}
        if use_vector_backend:
            query_embedding = self._get_embedding(retrieval_query, embedding_model)
            if query_embedding is not None:
                for entry in entries:
                    entry_embedding = self._get_embedding(self._entry_retrieval_text(entry), embedding_model)
                    if entry_embedding is None:
                        continue
                    score = self._cosine_similarity(query_embedding, entry_embedding)
                    if speaker and entry.speaker and entry.speaker.lower() == speaker.lower():
                        score += 0.1
                    if topic and entry.topic and entry.topic.lower() == topic.lower():
                        score += 0.1
                    if score <= 0.0:
                        continue
                    results.append((float(score), entry))
                if results:
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

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            documents = [self._entry_retrieval_text(entry) for entry in entries]
            vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english")
            matrix = vectorizer.fit_transform(documents + [retrieval_query])
            query_vec = matrix[-1]
            doc_matrix = matrix[:-1]
            similarities = (query_vec @ doc_matrix.T).toarray()[0]

            for entry, similarity in zip(entries, similarities):
                score = float(similarity)
                if speaker and entry.speaker and entry.speaker.lower() == speaker.lower():
                    score += 0.1
                if topic and entry.topic and entry.topic.lower() == topic.lower():
                    score += 0.1
                if score <= 0.0:
                    continue
                results.append((score, entry))
        except Exception:
            query_tokens = set(self._tokenize(retrieval_query))
            for entry in entries:
                entry_tokens = set(self._tokenize(self._entry_retrieval_text(entry)))
                overlap = len(query_tokens & entry_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(1, len(query_tokens))
                if speaker and entry.speaker and entry.speaker.lower() == speaker.lower():
                    score += 0.1
                if topic and entry.topic and entry.topic.lower() == topic.lower():
                    score += 0.1
                results.append((float(score), entry))

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
            parent_path = op.get("parent_path")
            source_path = op.get("source_path")
            source_paths = op.get("source_paths", [])
            subtopics = op.get("subtopics", [])
            assignments = op.get("assignments")

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
            elif event in ("CREATE_SUBTOPIC", "CREATE_TOPIC", "CREATE_NODE"):
                create_path = path or ((new_bank._normalize_path(parent_path) + [text]) if parent_path and text else parent_path)
                new_bank._create_subtopic(create_path, parent_path=parent_path)
            elif event in ("MOVE", "MOVE_NODE"):
                if entry_id and entry_id in new_bank._entries and path:
                    new_bank._move_entry_to_path(entry_id, path)
                elif source_path and path:
                    new_bank._move_subtree(source_path, path)
            elif event == "SPLIT_TOPIC":
                split_path = path or source_path
                if split_path and subtopics:
                    normalized_assignments = assignments if isinstance(assignments, dict) else None
                    new_bank._split_topic(split_path, [str(subtopic) for subtopic in subtopics], normalized_assignments)
            elif event == "MERGE_TOPIC":
                merge_target = path or parent_path
                normalized_sources = [sp for sp in source_paths if sp]
                if source_path:
                    normalized_sources.append(source_path)
                if merge_target and normalized_sources:
                    new_bank._merge_topics(normalized_sources, merge_target)
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
