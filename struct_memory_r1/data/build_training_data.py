"""
StructMemoryR1 training-data construction.

Builds parquet files consumed by veRL GRPO training for the three agents:

- Answer Agent (``--stage answer_agent``): question + speaker-grouped retrieved
  memories, scored against the gold short answer.
- Retrieve Agent (``--stage retrieve_agent``): question + tree schema, scored
  through the frozen Answer Agent after a deterministic plan execution.
- Memory Manager (``--stage memory_manager``): dialogue turn + current memory
  tree + new facts, scored through Retrieve Agent + Answer Agent on the
  updated tree.

Fact extraction has two modes:
1. LLM extraction via the OpenAI API (paper default; requires ``OPENAI_API_KEY``)
2. Observation extraction from LoCoMo's pre-extracted per-session observations
   (fallback when no API key is available)
"""
import json
import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from struct_memory_r1.data.locomo_loader import (
    Conversation, DialogueTurn, QAPair, Session,
    load_locomo, get_all_turns_flat, get_all_observations,
    split_locomo_train_val_test,
)
from struct_memory_r1.memory_bank import MemoryBank
from struct_memory_r1.prompts import (
    MEMORY_MANAGER_SYSTEM,
    ANSWER_AGENT_SYSTEM,
    RETRIEVER_AGENT_SYSTEM,
    make_memory_manager_training_prompt,
    make_answer_agent_training_prompt,
    make_retrieve_agent_training_prompt,
)


# ============================================================================
# Fact extraction
# ============================================================================

_HEURISTIC_STOPWORDS = (
    "hi ", "hey ", "hello", "oh ", "wow", "haha", "lol",
    "yeah", "yes", "no ", "ok", "sure",
)


def extract_facts_from_turn_heuristic(turn: DialogueTurn) -> List[str]:
    """Sentence-splitter fallback for LLM-free fact extraction."""
    import re
    text = (turn.text or "").strip()
    if not text:
        return []
    facts: List[str] = []
    for sentence in re.split(r"[.!?]+", text):
        sentence = sentence.strip()
        if len(sentence) > 15 and not sentence.lower().startswith(_HEURISTIC_STOPWORDS):
            facts.append(f"{turn.speaker}: {sentence}")
    return facts


def extract_facts_from_turn(
    turn: DialogueTurn,
    use_llm: bool = False,
    llm_model: str = "gpt-5-nano",
) -> List[str]:
    """Extract facts from a turn, optionally via the paper's LLM extractor."""
    if use_llm:
        try:
            from struct_memory_r1.llm_extract import extract_facts_from_turn_llm
            facts = extract_facts_from_turn_llm(
                speaker=turn.speaker, text=turn.text, model=llm_model,
            )
            if facts:
                return facts
        except (ImportError, ValueError) as exc:
            print(f"Warning: LLM extraction unavailable ({exc}); using heuristic.")
    return extract_facts_from_turn_heuristic(turn)


def extract_facts_from_observations(
    conversation: Conversation,
    session_idx: int,
) -> List[str]:
    """Use LoCoMo's per-session gold observations as facts."""
    if 0 <= session_idx < len(conversation.sessions):
        return list(conversation.sessions[session_idx].observations)
    return []


# ============================================================================
# Temporal memory bank construction
# ============================================================================

def build_temporal_memory_bank_from_turns(
    turns: List[DialogueTurn],
    max_previous_turns: int = 50,
    use_llm: bool = False,
    llm_model: str = "gpt-5-nano",
) -> Tuple[MemoryBank, List[str]]:
    """Build a memory tree by extracting facts from each turn in order."""
    bank = MemoryBank()
    all_facts: List[str] = []
    recent = turns[-max_previous_turns:] if len(turns) > max_previous_turns else turns
    for turn in recent:
        for fact in extract_facts_from_turn(turn, use_llm=use_llm, llm_model=llm_model):
            bank.insert_interaction(interaction=fact, speaker=turn.speaker)
            all_facts.append(fact)
    return bank, all_facts


def build_temporal_memory_bank(
    conversation: Conversation,
    up_to_session: int,
) -> MemoryBank:
    """Build a memory tree from gold observations up to ``up_to_session`` (1-based)."""
    bank = MemoryBank()
    for session in conversation.sessions[:up_to_session]:
        for obs in session.observations:
            speaker = _attribute_speaker(obs, conversation)
            bank.insert_interaction(
                interaction=obs, speaker=speaker, timestamp=session.datetime,
            )
    return bank


def _attribute_speaker(observation: str, conv: Conversation) -> Optional[str]:
    obs_lower = observation.lower()
    if conv.speaker_a.lower() in obs_lower:
        return conv.speaker_a
    if conv.speaker_b.lower() in obs_lower:
        return conv.speaker_b
    return None


# ============================================================================
# QA pair linking
# ============================================================================

def _parse_evidence_id(evidence_id: str) -> Optional[Tuple[int, int]]:
    """Parse a LoCoMo evidence id like ``D3:11`` into ``(session, turn)``."""
    try:
        clean = str(evidence_id).lstrip("D")
        parts = clean.split(":")
        session = int(parts[0])
        turn = int(parts[1]) if len(parts) > 1 else 1
        return session, turn
    except (ValueError, IndexError):
        return None


def get_qa_pairs_up_to_turn(
    conversation: Conversation,
    session_idx: int,
    turn_idx: int,
) -> List[QAPair]:
    """QA pairs whose evidence falls in or before ``(session_idx, turn_idx)`` (1-based)."""
    relevant: List[QAPair] = []
    for qa in conversation.qa_pairs:
        for ev_id in qa.evidence:
            parsed = _parse_evidence_id(ev_id)
            if parsed is None:
                continue
            ev_session, ev_turn = parsed
            if ev_session < session_idx or (ev_session == session_idx and ev_turn <= turn_idx):
                relevant.append(qa)
                break
    return relevant


def get_qa_pairs_for_session(
    conversation: Conversation,
    session_idx: int,
) -> List[QAPair]:
    """QA pairs whose evidence falls in or before ``session_idx`` (1-based)."""
    relevant: List[QAPair] = []
    for qa in conversation.qa_pairs:
        for ev_id in qa.evidence:
            parsed = _parse_evidence_id(ev_id)
            if parsed and parsed[0] <= session_idx:
                relevant.append(qa)
                break
    return relevant


# ============================================================================
# TF-IDF retrieval (for Answer Agent training data)
# ============================================================================

def tfidf_retrieve(query: str, candidates: List[str], topk: int = 30) -> List[str]:
    """Top-k candidates by TF-IDF cosine similarity to ``query``."""
    if not candidates:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english")
    matrix = vectorizer.fit_transform(candidates + [query])
    query_vec = matrix[-1]
    cand_matrix = matrix[:-1]
    scores = (query_vec @ cand_matrix.T).toarray()[0]
    top_indices = np.argsort(scores)[::-1][:topk]
    return [candidates[i] for i in top_indices]


# ============================================================================
# Struct-LoCoMo tree loading
# ============================================================================

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_struct_tree(
    sample_id: str,
    trees_dir: Optional[str] = None,
) -> Tuple[MemoryBank, Dict[str, str]]:
    """
    Load a Struct-LoCoMo XML tree for one conversation.

    Returns ``(bank, dia_id_to_entry_id)`` — the populated memory tree and a
    mapping from each observation's source dialogue id to its entry id in
    the bank, used to compute gold evidence for the Retrieve Agent.
    """
    if trees_dir is None:
        trees_dir = os.path.join(_repo_root(), "structured_locomo_trees")

    # Tree files follow the naming convention conv-<id>.xml (e.g. conv-26.xml).
    sample_token = sample_id.replace("conv-", "")
    path = os.path.join(trees_dir, f"conv-{sample_token}.xml")
    bank = MemoryBank()
    dia_to_entry: Dict[str, str] = {}
    if not os.path.isfile(path):
        return bank, dia_to_entry

    tree = ET.parse(path)
    root = tree.getroot()
    for speaker_node in root.iter("Speaker"):
        speaker = speaker_node.attrib.get("name") or "global"
        for session_node in speaker_node.iter("Session"):
            datetime = session_node.attrib.get("datetime")
            for obs_node in session_node.iter("Observation"):
                text_node = obs_node.find("text")
                dia_node = obs_node.find("source_dia_id")
                text = (text_node.text or "").strip() if text_node is not None else ""
                dia_id = (dia_node.text or "").strip() if dia_node is not None else ""
                if not text:
                    continue
                inserted = bank.insert_interaction(
                    interaction=text, speaker=speaker, timestamp=datetime,
                )
                if inserted and dia_id:
                    dia_to_entry[dia_id] = inserted[0].id
    return bank, dia_to_entry


# ============================================================================
# Memory Manager training data
# ============================================================================

def build_memory_manager_data(
    conversations: List[Conversation],
    split: str = "train",
    use_llm: bool = False,
    llm_model: str = "gpt-5-nano",
    max_previous_turns: int = 50,
) -> List[dict]:
    """
    Per-turn training samples for the Memory Manager.

    For each substantive dialogue turn:
    1. Snapshot the temporal memory tree built from previous turns/observations.
    2. Extract new facts from the current turn.
    3. Link QA pairs answerable as of this turn for the downstream reward.
    """
    samples: List[dict] = []
    idx = 0

    for conv in conversations:
        flat_turns: List[Dict] = []
        for s_idx, session in enumerate(conv.sessions):
            for t_idx, turn in enumerate(session.turns):
                flat_turns.append({
                    "turn": turn,
                    "session_idx": s_idx + 1,
                    "turn_idx": t_idx + 1,
                    "session_datetime": session.datetime,
                })

        for global_idx, info in enumerate(flat_turns):
            turn: DialogueTurn = info["turn"]
            session_idx: int = info["session_idx"]
            turn_idx: int = info["turn_idx"]

            if len(turn.text.strip()) < 10:
                continue

            previous_turns = [t["turn"] for t in flat_turns[:global_idx]]
            if use_llm and previous_turns:
                memory_bank, _ = build_temporal_memory_bank_from_turns(
                    previous_turns,
                    max_previous_turns=max_previous_turns,
                    use_llm=use_llm,
                    llm_model=llm_model,
                )
            else:
                memory_bank = build_temporal_memory_bank(conv, up_to_session=session_idx - 1)

            old_memory = memory_bank.to_structured_list()
            facts = extract_facts_from_turn(turn, use_llm=use_llm, llm_model=llm_model)
            if not facts:
                continue

            qa_targets = [
                {"question": qa.question, "answer": qa.answer}
                for qa in get_qa_pairs_up_to_turn(conv, session_idx, turn_idx)
            ]

            prompt_content = make_memory_manager_training_prompt(
                dialogue_turn=f"{turn.speaker}: {turn.text}",
                old_memory=old_memory,
                retrieved_facts=facts,
            )

            samples.append({
                "data_source": "struct_memory_manager",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "memory-management",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": qa_targets or [{"question": "", "answer": ""}],
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "conversation_id": conv.sample_id,
                    "session_idx": session_idx,
                    "turn_idx": turn_idx,
                    "old_memory": json.dumps(old_memory),
                    "facts": json.dumps(facts),
                },
            })
            idx += 1

    return samples


# ============================================================================
# Answer Agent training data
# ============================================================================

def build_answer_agent_data(
    conversations: List[Conversation],
    memories_per_speaker: int = 30,
    split: str = "train",
) -> List[dict]:
    """One Answer-Agent sample per question with 30 retrieved memories per speaker."""
    samples: List[dict] = []
    idx = 0

    for conv in conversations:
        bank = MemoryBank()
        for session in conv.sessions:
            for obs in session.observations:
                speaker = _attribute_speaker(obs, conv)
                if speaker is None:
                    # Ambiguous observation — surface it under both speakers.
                    bank.insert_interaction(obs, speaker=conv.speaker_a, timestamp=session.datetime)
                    bank.insert_interaction(obs, speaker=conv.speaker_b, timestamp=session.datetime)
                else:
                    bank.insert_interaction(obs, speaker=speaker, timestamp=session.datetime)

        for qa in conv.qa_pairs:
            retrieved_a = bank.retrieve(qa.question, topk=memories_per_speaker, speaker=conv.speaker_a)
            retrieved_b = bank.retrieve(qa.question, topk=memories_per_speaker, speaker=conv.speaker_b)
            mems_a = [
                f"{m['timestamp']}: {m['text']}" if m.get("timestamp") else m["text"]
                for m in retrieved_a
            ]
            mems_b = [
                f"{m['timestamp']}: {m['text']}" if m.get("timestamp") else m["text"]
                for m in retrieved_b
            ]

            prompt_content = make_answer_agent_training_prompt(
                question=qa.question,
                memories_speaker_a=mems_a,
                memories_speaker_b=mems_b,
                speaker_a_name=conv.speaker_a,
                speaker_b_name=conv.speaker_b,
            )

            samples.append({
                "data_source": "struct_answer_agent",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "memory-qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": [qa.answer]},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "conversation_id": conv.sample_id,
                    "category": qa.category,
                },
            })
            idx += 1

    return samples


# ============================================================================
# Retrieve Agent training data
# ============================================================================

def build_retrieve_agent_data(
    conversations: List[Conversation],
    split: str = "train",
    trees_dir: Optional[str] = None,
) -> List[dict]:
    """
    One Retrieve-Agent sample per question.

    The Retrieve Agent sees the fixed Struct-LoCoMo tree schema and the
    question; its plan is executed deterministically against the tree, and
    the reward is the frozen Answer Agent's score on the retrieved leaves.
    """
    samples: List[dict] = []
    idx = 0

    for conv in conversations:
        bank, dia_to_entry = load_struct_tree(conv.sample_id, trees_dir=trees_dir)
        # Fallback: if the conversation has no XML tree, build one from observations.
        if len(bank) == 0:
            bank = build_temporal_memory_bank(conv, up_to_session=len(conv.sessions))

        structured_memory = bank.to_structured_list()
        schema = bank.schema_summary() if hasattr(bank, "schema_summary") else {}

        for qa in conv.qa_pairs:
            gold_entry_ids = sorted({
                dia_to_entry[ev] for ev in qa.evidence if ev in dia_to_entry
            })
            prompt_content = make_retrieve_agent_training_prompt(
                question=qa.question,
                schema=schema,
            )

            samples.append({
                "data_source": "struct_retriever_agent",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "memory-retrieval",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": [{"question": qa.question, "answer": qa.answer}],
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "conversation_id": conv.sample_id,
                    "question": qa.question,
                    "category": qa.category,
                    "old_memory": json.dumps(structured_memory),
                    "gold_entry_ids": json.dumps(gold_entry_ids),
                },
            })
            idx += 1

    return samples


# ============================================================================
# Parquet I/O
# ============================================================================

def save_as_parquet(samples: List[dict], filepath: str) -> None:
    """Persist training samples as a parquet file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(samples).to_parquet(filepath, index=False)
    print(f"Saved {len(samples)} samples to {filepath}")


# ============================================================================
# CLI entry point
# ============================================================================

_STAGE_BUILDERS = {
    "answer_agent": "answer_agent",
    "memory_manager": "memory_manager",
    "retrieve_agent": "retrieve_agent",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build StructMemoryR1 training data")
    parser.add_argument("--stage", default="all",
                        choices=["answer_agent", "memory_manager", "retrieve_agent", "all"])
    parser.add_argument("--use_llm", action="store_true",
                        help="Use the LLM fact extractor (requires OPENAI_API_KEY).")
    parser.add_argument("--llm_model", default="gpt-5-nano",
                        help="OpenAI model name for fact extraction.")
    parser.add_argument("--data_dir", default="data/locomo",
                        help="Directory containing locomo10.json.")
    parser.add_argument("--output_dir", default="data/struct_memory_r1",
                        help="Directory to write training parquet files.")
    parser.add_argument("--trees_dir", default=None,
                        help="Override for the Struct-LoCoMo XML tree directory.")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    print("Loading LoCoMo dataset...")
    conversations = load_locomo(args.data_dir)
    print(f"Loaded {len(conversations)} conversations")

    train, val, test = split_locomo_train_val_test(conversations)
    print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test conversations")

    stages = (
        ["answer_agent", "retrieve_agent", "memory_manager"]
        if args.stage == "all" else [args.stage]
    )

    if "memory_manager" in stages:
        print("\nBuilding Memory Manager training data...")
        mm_train = build_memory_manager_data(train, split="train",
                                             use_llm=args.use_llm, llm_model=args.llm_model)
        mm_val = build_memory_manager_data(val, split="val",
                                           use_llm=args.use_llm, llm_model=args.llm_model)
        mm_test = build_memory_manager_data(test, split="test",
                                            use_llm=args.use_llm, llm_model=args.llm_model)
        save_as_parquet(mm_train + mm_val, f"{args.output_dir}/memory_manager/train.parquet")
        save_as_parquet(mm_test, f"{args.output_dir}/memory_manager/test.parquet")

    if "answer_agent" in stages:
        print("\nBuilding Answer Agent training data...")
        aa_train = build_answer_agent_data(train, split="train")
        aa_val = build_answer_agent_data(val, split="val")
        aa_test = build_answer_agent_data(test, split="test")
        save_as_parquet(aa_train + aa_val, f"{args.output_dir}/answer_agent/train.parquet")
        save_as_parquet(aa_test, f"{args.output_dir}/answer_agent/test.parquet")

    if "retrieve_agent" in stages:
        print("\nBuilding Retrieve Agent training data...")
        ra_train = build_retrieve_agent_data(train, split="train", trees_dir=args.trees_dir)
        ra_val = build_retrieve_agent_data(val, split="val", trees_dir=args.trees_dir)
        ra_test = build_retrieve_agent_data(test, split="test", trees_dir=args.trees_dir)
        save_as_parquet(ra_train + ra_val, f"{args.output_dir}/retrieve_agent/train.parquet")
        save_as_parquet(ra_test, f"{args.output_dir}/retrieve_agent/test.parquet")

    print(f"\nDone. Data written to {args.output_dir}/")


if __name__ == "__main__":
    main()
