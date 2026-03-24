"""
Data construction for Memory-R1 training.

Implements Algorithms 1 and 2 from the Memory-R1 paper:
- Algorithm 1: Memory Manager training data
  For each dialogue TURN t, build a temporal memory bank from the previous
  turns (up to 50) using LLM extraction (GPT-5-nano), combine with the
  current turn and linked QA pairs.
- Algorithm 2: Answer Agent training data
  For each question q, retrieve 30 memories per speaker (60 total) via
  similarity-based retrieval, paired with the gold answer.

Fact extraction modes:
1. LLM-based extraction via OpenAI API (paper's approach, requires OPENAI_API_KEY)
2. Observation-based extraction using LoCoMo's pre-extracted observations (fallback)
"""
import json
import os
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from memory_r1.data.locomo_loader import (
    Conversation, Session, DialogueTurn, QAPair,
    load_locomo, get_all_turns_flat, get_all_observations,
    split_locomo_train_val_test,
)
from memory_r1.memory_bank import MemoryBank
from memory_r1.prompts import (
    make_memory_manager_training_prompt,
    make_answer_agent_training_prompt,
    MEMORY_MANAGER_SYSTEM,
    ANSWER_AGENT_SYSTEM,
)


# ============================================================================
# Fact extraction
# ============================================================================

def extract_facts_from_turn_heuristic(turn: DialogueTurn) -> List[str]:
    """
    Heuristic fact extraction from a dialogue turn.
    Fallback when LLM extraction is unavailable.
    """
    import re
    text = turn.text.strip()
    if not text:
        return []

    sentences = re.split(r'[.!?]+', text)
    facts = []
    for s in sentences:
        s = s.strip()
        if len(s) > 15 and not s.lower().startswith(
            ("hi ", "hey ", "hello", "oh ", "wow", "haha", "lol", "yeah", "yes", "no ", "ok", "sure")
        ):
            facts.append(f"{turn.speaker}: {s}")
    return facts


def extract_facts_from_turn(
    turn: DialogueTurn,
    use_llm: bool = False,
    llm_model: str = "gpt-5-nano",
) -> List[str]:
    """
    Extract facts from a dialogue turn.

    Args:
        turn: The dialogue turn
        use_llm: Whether to use LLM-based extraction (paper's approach)
        llm_model: Which OpenAI model to use for extraction

    Returns:
        List of extracted fact strings
    """
    if use_llm:
        try:
            from memory_r1.llm_extract import extract_facts_from_turn_llm
            facts = extract_facts_from_turn_llm(
                speaker=turn.speaker,
                text=turn.text,
                model=llm_model,
            )
            if facts:
                return facts
        except (ImportError, ValueError) as e:
            print(f"Warning: LLM extraction unavailable ({e}), falling back to heuristic")

    return extract_facts_from_turn_heuristic(turn)


def extract_facts_from_observations(
    conversation: Conversation,
    session_idx: int,
) -> List[str]:
    """
    Use LoCoMo's pre-extracted observations as facts.
    These are speaker-attributed atomic claims extracted per session.
    """
    if session_idx < len(conversation.sessions):
        return list(conversation.sessions[session_idx].observations)
    return []


# ============================================================================
# Temporal memory bank construction (Algorithm 1)
# ============================================================================

def build_temporal_memory_bank_from_turns(
    turns: List[DialogueTurn],
    max_previous_turns: int = 50,
    use_llm: bool = False,
    llm_model: str = "gpt-5-nano",
) -> Tuple[MemoryBank, List[str]]:
    """
    Build a temporal memory bank from previous turns using fact extraction.

    Following Algorithm 1: for each turn, extract facts and add them to
    the memory bank. Uses up to the most recent max_previous_turns.

    Args:
        turns: List of dialogue turns to process
        max_previous_turns: Maximum number of previous turns to consider (paper: 50)
        use_llm: Whether to use LLM extraction
        llm_model: OpenAI model for LLM extraction

    Returns:
        (memory_bank, all_extracted_facts): The built memory bank and all facts
    """
    bank = MemoryBank()
    all_facts = []

    # Only consider the most recent max_previous_turns
    recent_turns = turns[-max_previous_turns:] if len(turns) > max_previous_turns else turns

    for turn in recent_turns:
        facts = extract_facts_from_turn(turn, use_llm=use_llm, llm_model=llm_model)
        for fact in facts:
            bank.add(fact)
            all_facts.append(fact)

    return bank, all_facts


def build_temporal_memory_bank(
    conversation: Conversation,
    up_to_session: int,
) -> MemoryBank:
    """
    Build a temporal memory bank from observations up to a given session.
    Observation-based fallback for when LLM extraction isn't used.
    """
    bank = MemoryBank()
    for session in conversation.sessions[:up_to_session]:
        for obs in session.observations:
            bank.add(obs)
    return bank


# ============================================================================
# QA pair linking
# ============================================================================

def get_qa_pairs_up_to_turn(
    conversation: Conversation,
    session_idx: int,
    turn_idx: int,
) -> List[QAPair]:
    """
    Get QA pairs whose evidence falls within or before a given turn.

    Args:
        conversation: The conversation
        session_idx: Current session index (1-based)
        turn_idx: Turn index within the session

    Returns:
        List of relevant QA pairs
    """
    relevant_qas = []
    for qa in conversation.qa_pairs:
        is_relevant = False
        for evidence_id in qa.evidence:
            try:
                clean = evidence_id.lstrip("D")
                parts = clean.split(":")
                ev_session = int(parts[0])
                ev_turn = int(parts[1]) if len(parts) > 1 else 1
                # Evidence is from a session before current, or from current
                # session but at or before the current turn
                if ev_session < session_idx:
                    is_relevant = True
                elif ev_session == session_idx and ev_turn <= turn_idx:
                    is_relevant = True
                if is_relevant:
                    break
            except (ValueError, IndexError):
                continue
        if is_relevant:
            relevant_qas.append(qa)
    return relevant_qas


def get_qa_pairs_for_session(
    conversation: Conversation,
    session_idx: int,
) -> List[QAPair]:
    """
    Get QA pairs whose evidence falls within or before a given session.
    """
    relevant_qas = []
    for qa in conversation.qa_pairs:
        for evidence_id in qa.evidence:
            try:
                clean = evidence_id.lstrip("D")
                ev_session = int(clean.split(":")[0])
                if ev_session <= session_idx:
                    relevant_qas.append(qa)
                    break
            except (ValueError, IndexError):
                continue
    return relevant_qas


# ============================================================================
# TF-IDF retrieval for Answer Agent data construction
# ============================================================================

def tfidf_retrieve(
    query: str,
    candidates: List[str],
    topk: int = 30,
) -> List[str]:
    """
    Retrieve top-k candidates by TF-IDF similarity to the query.

    This approximates the paper's "similarity-based RAG" retrieval
    (Section 3.2) without requiring a dense embedding model.

    Args:
        query: The search query
        candidates: List of candidate memory strings
        topk: Number of results

    Returns:
        Top-k most similar candidates
    """
    if not candidates:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Fit on candidates + query
    all_texts = candidates + [query]
    vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Query is the last vector
    query_vec = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]

    # Compute cosine similarity
    scores = (query_vec @ candidate_matrix.T).toarray()[0]
    top_indices = np.argsort(scores)[::-1][:topk]

    return [candidates[i] for i in top_indices]


# ============================================================================
# Memory Manager training data (Algorithm 1 - per-turn)
# ============================================================================

def build_memory_manager_data(
    conversations: List[Conversation],
    split: str = "train",
    use_llm: bool = False,
    llm_model: str = "gpt-5-nano",
    max_previous_turns: int = 50,
) -> List[dict]:
    """
    Build training data for the Memory Manager (Algorithm 1).

    Per the paper: for each dialogue TURN t:
    1. Build temporal memory bank from the previous turns (up to 50)
       using LLM extraction (or observations as fallback)
    2. Extract facts from the current turn
    3. Pair with any QA pairs answerable up to this turn
    4. Store as training tuple: (dialogue_turn, temporal_memory_bank, QA)

    Args:
        conversations: List of parsed LoCoMo conversations
        split: Data split label ("train", "val", "test")
        use_llm: Whether to use LLM fact extraction
        llm_model: OpenAI model for LLM extraction
        max_previous_turns: Max previous turns for memory bank (paper: 50)

    Returns:
        List of training samples in veRL parquet format
    """
    samples = []
    idx = 0

    for conv in conversations:
        # Collect all turns across all sessions with their session info
        all_turns_with_info = []
        for s_idx, session in enumerate(conv.sessions):
            for t_idx, turn in enumerate(session.turns):
                all_turns_with_info.append({
                    "turn": turn,
                    "session_idx": s_idx + 1,  # 1-based
                    "turn_idx": t_idx + 1,  # 1-based within session
                    "session_datetime": session.datetime,
                })

        # Process each turn
        for global_t_idx, turn_info in enumerate(all_turns_with_info):
            turn = turn_info["turn"]
            session_idx = turn_info["session_idx"]
            turn_idx = turn_info["turn_idx"]

            # Skip very short turns (greetings, etc.)
            if len(turn.text.strip()) < 10:
                continue

            # 1. Build temporal memory bank from PREVIOUS turns
            previous_turns = [ti["turn"] for ti in all_turns_with_info[:global_t_idx]]
            if use_llm and previous_turns:
                memory_bank, _ = build_temporal_memory_bank_from_turns(
                    previous_turns,
                    max_previous_turns=max_previous_turns,
                    use_llm=use_llm,
                    llm_model=llm_model,
                )
            else:
                # Fallback: use observations from previous sessions
                memory_bank = build_temporal_memory_bank(conv, up_to_session=session_idx - 1)

            old_memory = memory_bank.to_list()

            # 2. Extract facts from the CURRENT turn
            facts = extract_facts_from_turn(turn, use_llm=use_llm, llm_model=llm_model)
            if not facts:
                continue

            # 3. Get linked QA pairs for reward computation
            linked_qas = get_qa_pairs_up_to_turn(conv, session_idx, turn_idx)

            # Format the current dialogue turn text
            turn_text = f"{turn.speaker}: {turn.text}"

            # Create the training prompt
            prompt_content = make_memory_manager_training_prompt(
                dialogue_turn=turn_text,
                old_memory=old_memory,
                retrieved_facts=facts,
            )

            # Store QA pairs for reward computation during training
            qa_targets = [{"question": qa.question, "answer": qa.answer}
                          for qa in linked_qas]

            sample = {
                "data_source": "memory_manager",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "memory-management",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": qa_targets if qa_targets else [{"question": "", "answer": ""}],
                    }
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "conversation_id": conv.sample_id,
                    "session_idx": session_idx,
                    "turn_idx": turn_idx,
                    "old_memory": json.dumps(old_memory),
                    "facts": json.dumps(facts),
                }
            }
            samples.append(sample)
            idx += 1

    return samples


# ============================================================================
# Answer Agent training data (Algorithm 2 - with speaker-grouped retrieval)
# ============================================================================

def build_answer_agent_data(
    conversations: List[Conversation],
    memories_per_speaker: int = 30,
    split: str = "train",
) -> List[dict]:
    """
    Build training data for the Answer Agent (Algorithm 2).

    Per the paper: for each question q:
    1. Build complete memory bank from all observations
    2. Retrieve top-30 most relevant memories per speaker via
       similarity-based retrieval (60 total)
    3. Format with speaker grouping and timestamps (Figure 11)
    4. Pair with gold answer

    Args:
        conversations: List of parsed LoCoMo conversations
        memories_per_speaker: Memories to retrieve per speaker (paper: 30)
        split: Data split label

    Returns:
        List of training samples in veRL parquet format
    """
    samples = []
    idx = 0

    for conv in conversations:
        # Build observation pools per speaker, with timestamps
        # Each observation is attributed to a session with a datetime
        memories_pool_a = []  # (text_with_timestamp, text_only)
        memories_pool_b = []

        for session in conv.sessions:
            timestamp = session.datetime or ""
            for obs in session.observations:
                # Attribute observation to speaker based on name mention
                obs_lower = obs.lower()
                # Format with timestamp as in Figure 11
                timestamped = f"{timestamp}: {obs}" if timestamp else obs

                if conv.speaker_a.lower() in obs_lower:
                    memories_pool_a.append((timestamped, obs))
                elif conv.speaker_b.lower() in obs_lower:
                    memories_pool_b.append((timestamped, obs))
                else:
                    # Can't determine speaker; add to both
                    memories_pool_a.append((timestamped, obs))
                    memories_pool_b.append((timestamped, obs))

        for qa in conv.qa_pairs:
            # Retrieve top-k per speaker using TF-IDF similarity
            # (approximates the paper's embedding-based RAG)
            raw_a = [text for _, text in memories_pool_a]
            raw_b = [text for _, text in memories_pool_b]

            # Retrieve most relevant memories per speaker
            if raw_a:
                retrieved_a_texts = tfidf_retrieve(qa.question, raw_a, topk=memories_per_speaker)
                # Map back to timestamped versions
                text_to_ts = {text: ts for ts, text in memories_pool_a}
                retrieved_a = [text_to_ts.get(t, t) for t in retrieved_a_texts]
            else:
                retrieved_a = []

            if raw_b:
                retrieved_b_texts = tfidf_retrieve(qa.question, raw_b, topk=memories_per_speaker)
                text_to_ts = {text: ts for ts, text in memories_pool_b}
                retrieved_b = [text_to_ts.get(t, t) for t in retrieved_b_texts]
            else:
                retrieved_b = []

            # Create training prompt with speaker-grouped memories
            prompt_content = make_answer_agent_training_prompt(
                question=qa.question,
                memories_speaker_a=retrieved_a,
                memories_speaker_b=retrieved_b,
                speaker_a_name=conv.speaker_a,
                speaker_b_name=conv.speaker_b,
            )

            sample = {
                "data_source": "answer_agent",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "memory-qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": [qa.answer],
                    }
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "conversation_id": conv.sample_id,
                    "category": qa.category,
                }
            }
            samples.append(sample)
            idx += 1

    return samples


# ============================================================================
# Parquet I/O
# ============================================================================

def save_as_parquet(samples: List[dict], filepath: str):
    """Save training samples as parquet file for veRL."""
    df = pd.DataFrame(samples)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(samples)} samples to {filepath}")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Build all Memory-R1 training data from LoCoMo."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Memory-R1 training data")
    parser.add_argument("--use_llm", action="store_true",
                        help="Use LLM (GPT-5-nano) for fact extraction (requires OPENAI_API_KEY)")
    parser.add_argument("--llm_model", default="gpt-5-nano",
                        help="OpenAI model for fact extraction")
    parser.add_argument("--data_dir", default="data/locomo",
                        help="Directory for LoCoMo data")
    parser.add_argument("--output_dir", default="data/memory_r1",
                        help="Output directory for training data")
    args = parser.parse_args()

    random.seed(42)

    print("Loading LoCoMo dataset...")
    conversations = load_locomo(args.data_dir)
    print(f"Loaded {len(conversations)} conversations")

    # Split following paper: 1:1:8 ratio (152/81/1307 questions)
    train_convs, val_convs, test_convs = split_locomo_train_val_test(conversations)
    print(f"Split: {len(train_convs)} train, {len(val_convs)} val, {len(test_convs)} test")

    if args.use_llm:
        print(f"\nUsing LLM extraction with model: {args.llm_model}")
    else:
        print("\nUsing observation-based extraction (set --use_llm for LLM extraction)")

    # Build Memory Manager data (per-turn, Algorithm 1)
    print("\nBuilding Memory Manager training data (per-turn)...")
    mm_train = build_memory_manager_data(
        train_convs, split="train",
        use_llm=args.use_llm, llm_model=args.llm_model,
    )
    mm_val = build_memory_manager_data(
        val_convs, split="val",
        use_llm=args.use_llm, llm_model=args.llm_model,
    )
    mm_test = build_memory_manager_data(
        test_convs, split="test",
        use_llm=args.use_llm, llm_model=args.llm_model,
    )
    print(f"Memory Manager: {len(mm_train)} train, {len(mm_val)} val, {len(mm_test)} test")

    save_as_parquet(mm_train + mm_val, f"{args.output_dir}/memory_manager/train.parquet")
    save_as_parquet(mm_test, f"{args.output_dir}/memory_manager/test.parquet")

    # Build Answer Agent data (Algorithm 2)
    print("\nBuilding Answer Agent training data (speaker-grouped retrieval)...")
    aa_train = build_answer_agent_data(train_convs, split="train")
    aa_val = build_answer_agent_data(val_convs, split="val")
    aa_test = build_answer_agent_data(test_convs, split="test")
    print(f"Answer Agent: {len(aa_train)} train, {len(aa_val)} val, {len(aa_test)} test")

    save_as_parquet(aa_train + aa_val, f"{args.output_dir}/answer_agent/train.parquet")
    save_as_parquet(aa_test, f"{args.output_dir}/answer_agent/test.parquet")

    print(f"\nDone! Data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
