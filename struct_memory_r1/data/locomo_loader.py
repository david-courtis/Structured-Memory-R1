"""
LoCoMo dataset loader for StructMemoryR1.

Loads and parses the LoCoMo (Maharana et al., 2024) multi-session dialogue
benchmark from a local JSON file (or downloads it from GitHub when absent).
The structured-memory trees used at training time live alongside the JSON in
``structured_locomo_trees/``; see ``build_training_data.load_struct_trees``.

Dataset source: https://github.com/snap-research/locomo
"""
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DialogueTurn:
    speaker: str
    dia_id: str
    text: str
    img_url: Optional[str] = None
    blip_caption: Optional[str] = None


@dataclass
class Session:
    session_id: int
    datetime: Optional[str]
    turns: List[DialogueTurn]
    observations: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    events: List[Dict] = field(default_factory=list)


@dataclass
class QAPair:
    question: str
    answer: str
    category: str  # single-hop, multi-hop, temporal, open-domain, adversarial
    evidence: List[str]  # list of dia_ids


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: List[Session]
    qa_pairs: List[QAPair]


# Category id → name (LoCoMo's qa.category is an integer).
CATEGORY_MAP = {
    1: "single-hop", 2: "multi-hop", 3: "temporal",
    4: "open-domain", 5: "adversarial",
}


def _find_locomo_json(data_dir: str) -> Optional[str]:
    """Look for a locomo10.json file in the usual locations."""
    candidates = [
        os.path.join(data_dir, "locomo10.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "structured_locomo_trees", "locomo10.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def download_locomo(save_dir: str = "data/locomo") -> str:
    """Return the path to locomo10.json, downloading it if necessary."""
    os.makedirs(save_dir, exist_ok=True)
    local = _find_locomo_json(save_dir)
    if local is not None:
        return local

    filepath = os.path.join(save_dir, "locomo10.json")
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    import urllib.request
    print(f"Downloading LoCoMo dataset to {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    return filepath


def parse_locomo(filepath: str) -> List[Conversation]:
    """Parse the LoCoMo JSON file into structured ``Conversation`` objects."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversations: List[Conversation] = []
    for item in raw_data:
        sample_id = str(item.get("sample_id", ""))

        speaker_a = item.get("speaker_a")
        speaker_b = item.get("speaker_b")
        if not speaker_a or not speaker_b:
            first_session = item.get("conversation", {}).get("session_1", [])
            seen: List[str] = []
            for turn in first_session:
                name = turn.get("speaker", "")
                if name and name not in seen:
                    seen.append(name)
                if len(seen) == 2:
                    break
            speaker_a = speaker_a or (seen[0] if len(seen) > 0 else "Speaker A")
            speaker_b = speaker_b or (seen[1] if len(seen) > 1 else "Speaker B")

        sessions: List[Session] = []
        idx = 1
        while f"session_{idx}" in item.get("conversation", {}):
            conv = item["conversation"]
            turns_raw = conv.get(f"session_{idx}", [])
            turns = [
                DialogueTurn(
                    speaker=t.get("speaker", ""),
                    dia_id=t.get("dia_id", ""),
                    text=t.get("text", ""),
                    img_url=t.get("img_url"),
                    blip_caption=t.get("blip_caption"),
                )
                for t in turns_raw
            ]

            raw_obs = item.get("observation", {}).get(f"session_{idx}_observation", [])
            observations: List[str] = []
            if isinstance(raw_obs, dict):
                for fact_list in raw_obs.values():
                    if isinstance(fact_list, list):
                        for entry in fact_list:
                            if isinstance(entry, list) and entry:
                                observations.append(entry[0])
                            elif isinstance(entry, str):
                                observations.append(entry)
            elif isinstance(raw_obs, list):
                for entry in raw_obs:
                    if isinstance(entry, str):
                        observations.append(entry)
                    elif isinstance(entry, list) and entry:
                        observations.append(entry[0])
            elif isinstance(raw_obs, str):
                observations = [raw_obs]

            summary = item.get("session_summary", {}).get(f"session_{idx}")
            events: List[Dict] = []
            ev = item.get("event_summary", {}).get(f"session_{idx}")
            if isinstance(ev, list):
                events = ev
            elif isinstance(ev, dict):
                events = [ev]

            sessions.append(Session(
                session_id=idx,
                datetime=conv.get(f"session_{idx}_date_time"),
                turns=turns,
                observations=observations,
                summary=summary,
                events=events,
            ))
            idx += 1

        qa_pairs: List[QAPair] = []
        for qa in item.get("qa", []):
            category = CATEGORY_MAP.get(qa.get("category"), str(qa.get("category", "")))
            # Skip adversarial questions per Memory-R1 evaluation protocol.
            if category == "adversarial":
                continue
            qa_pairs.append(QAPair(
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                category=category,
                evidence=qa.get("evidence", []),
            ))

        conversations.append(Conversation(
            sample_id=sample_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qa_pairs=qa_pairs,
        ))

    return conversations


def get_all_turns_flat(conversation: Conversation) -> List[DialogueTurn]:
    """All dialogue turns across all sessions, in chronological order."""
    return [t for s in conversation.sessions for t in s.turns]


def get_all_observations(conversation: Conversation) -> List[str]:
    """All observations across all sessions, in chronological order."""
    return [o for s in conversation.sessions for o in s.observations]


def split_locomo_train_val_test(
    conversations: List[Conversation],
    train_ratio: float = 0.125,
    val_ratio: float = 0.0625,
) -> Tuple[List[Conversation], List[Conversation], List[Conversation]]:
    """
    1:1:8 train/val/test split following Memory-R1.

    With LoCoMo's 10 conversations this produces 1 train / 1 val / 8 test,
    matching the paper's 152 / 81 / 1,307 QA-pair split.
    """
    n = len(conversations)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    return (
        conversations[:n_train],
        conversations[n_train:n_train + n_val],
        conversations[n_train + n_val:],
    )


def load_locomo(save_dir: str = "data/locomo") -> List[Conversation]:
    """Load LoCoMo conversations, downloading the dataset on cache miss."""
    return parse_locomo(download_locomo(save_dir))
