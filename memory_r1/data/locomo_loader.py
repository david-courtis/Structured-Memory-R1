"""
LoCoMo dataset loader.

Downloads and parses the LoCoMo dataset (Maharana et al., 2024) from GitHub.
The dataset contains 10 long multi-session dialogues with QA pairs,
observations, session summaries, and event summaries.

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


def download_locomo(save_dir: str = "data/locomo") -> str:
    """Download the LoCoMo dataset from GitHub."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "locomo10.json")

    if os.path.exists(filepath):
        return filepath

    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    try:
        import urllib.request
        print(f"Downloading LoCoMo dataset to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download LoCoMo: {e}") from e

    return filepath


def parse_locomo(filepath: str) -> List[Conversation]:
    """
    Parse the LoCoMo JSON file into structured Conversation objects.

    The LoCoMo format has sessions as session_1, session_2, etc.,
    with corresponding datetime, observations, summaries, and events.
    """
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    conversations = []
    for item in raw_data:
        sample_id = str(item.get("sample_id", ""))

        # Extract speaker names from first session's dialogue turns
        speaker_a = item.get("speaker_a", None)
        speaker_b = item.get("speaker_b", None)
        if not speaker_a or not speaker_b:
            first_session = item.get("conversation", {}).get("session_1", [])
            speakers_seen = []
            for t in first_session:
                s = t.get("speaker", "")
                if s and s not in speakers_seen:
                    speakers_seen.append(s)
                if len(speakers_seen) == 2:
                    break
            speaker_a = speakers_seen[0] if len(speakers_seen) > 0 else "Speaker A"
            speaker_b = speakers_seen[1] if len(speakers_seen) > 1 else "Speaker B"

        # Parse sessions
        sessions = []
        session_idx = 1
        while f"session_{session_idx}" in item.get("conversation", {}):
            conv = item["conversation"]
            session_key = f"session_{session_idx}"
            datetime_key = f"session_{session_idx}_date_time"

            turns_raw = conv.get(session_key, [])
            turns = []
            for t in turns_raw:
                turns.append(DialogueTurn(
                    speaker=t.get("speaker", ""),
                    dia_id=t.get("dia_id", ""),
                    text=t.get("text", ""),
                    img_url=t.get("img_url"),
                    blip_caption=t.get("blip_caption"),
                ))

            # Parse observations
            # LoCoMo format: {speaker_name: [[fact_text, dia_id], ...], ...}
            obs_key = f"session_{session_idx}_observation"
            raw_obs = item.get("observation", {}).get(obs_key, [])
            observations = []
            if isinstance(raw_obs, dict):
                for speaker_name, fact_list in raw_obs.items():
                    if isinstance(fact_list, list):
                        for entry in fact_list:
                            if isinstance(entry, list) and len(entry) >= 1:
                                observations.append(entry[0])
                            elif isinstance(entry, str):
                                observations.append(entry)
            elif isinstance(raw_obs, list):
                for entry in raw_obs:
                    if isinstance(entry, str):
                        observations.append(entry)
                    elif isinstance(entry, list) and len(entry) >= 1:
                        observations.append(entry[0])
            elif isinstance(raw_obs, str):
                observations = [raw_obs]

            # Parse summary
            summary_key = f"session_{session_idx}"
            summary = item.get("session_summary", {}).get(summary_key)

            # Parse events
            events = []
            event_data = item.get("event_summary", {})
            # Events are keyed by session, but format varies
            event_key = f"session_{session_idx}"
            if event_key in event_data:
                ev = event_data[event_key]
                if isinstance(ev, list):
                    events = ev
                elif isinstance(ev, dict):
                    events = [ev]

            sessions.append(Session(
                session_id=session_idx,
                datetime=conv.get(datetime_key),
                turns=turns,
                observations=observations,
                summary=summary,
                events=events,
            ))
            session_idx += 1

        # Parse QA pairs
        # LoCoMo category mapping: 1=single-hop, 2=multi-hop, 3=temporal,
        # 4=open-domain, 5=adversarial
        CATEGORY_MAP = {
            1: "single-hop", 2: "multi-hop", 3: "temporal",
            4: "open-domain", 5: "adversarial",
        }
        qa_pairs = []
        for qa in item.get("qa", []):
            raw_cat = qa.get("category", "")
            category = CATEGORY_MAP.get(raw_cat, str(raw_cat))
            # Skip adversarial questions as per Memory-R1 paper
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
    """Get all dialogue turns across all sessions in order."""
    turns = []
    for session in conversation.sessions:
        turns.extend(session.turns)
    return turns


def get_all_observations(conversation: Conversation) -> List[str]:
    """Get all observations across all sessions."""
    obs = []
    for session in conversation.sessions:
        obs.extend(session.observations)
    return obs


def split_locomo_train_val_test(
    conversations: List[Conversation],
    train_ratio: float = 0.125,  # ~1 conversation
    val_ratio: float = 0.0625,   # ~0.5 conversations
) -> Tuple[List, List, List]:
    """
    Split LoCoMo conversations into train/val/test following paper's
    1:1:8 split (152/81/1307 questions).

    With only 10 conversations, we use 1 for train, 1 for val, 8 for test.
    """
    n = len(conversations)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train = conversations[:n_train]
    val = conversations[n_train:n_train + n_val]
    test = conversations[n_train + n_val:]
    return train, val, test


def load_locomo(save_dir: str = "data/locomo") -> List[Conversation]:
    """Download (if needed) and load the LoCoMo dataset."""
    filepath = download_locomo(save_dir)
    return parse_locomo(filepath)
