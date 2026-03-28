"""
LLM-based fact extraction for Memory-R1.

The paper (Section B.2) uses GPT-4o-mini to extract key information from
dialogue turns. We use GPT-5-nano as the extraction model via the OpenAI API.

Set the OPENAI_API_KEY environment variable before use.
"""
import os
import json
import time
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


EXTRACT_SYSTEM_PROMPT = """You are a fact extraction system. Given a dialogue turn between two speakers, extract the key factual statements as a JSON list of strings.

Rules:
- Extract concrete facts, events, preferences, relationships, and plans
- Each fact should be a self-contained statement attributed to the speaker
- Ignore greetings, filler words, and purely social exchanges
- Keep each fact concise (one sentence)
- Include the speaker's name in each fact
- Return a JSON array of strings

Example input:
"Caroline: I just went to the LGBTQ support group meeting yesterday. It was really helpful."

Example output:
["Caroline attended an LGBTQ support group meeting.", "Caroline found the LGBTQ support group meeting helpful."]"""


_client = None


def _get_client() -> "OpenAI":
    """Lazily initialize the OpenAI client."""
    global _client
    if _client is None:
        if OpenAI is None:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )
        _client = OpenAI(api_key=api_key)
    return _client


def llm_extract_facts(
    turn_text: str,
    speaker: str = "",
    model: str = "gpt-5-nano",
    max_retries: int = 3,
) -> List[str]:
    """
    Extract key facts from a dialogue turn using an LLM.

    This replaces the paper's GPT-4o-mini extraction step.

    Args:
        turn_text: The dialogue turn text
        speaker: Speaker name (for attribution)
        model: OpenAI model to use
        max_retries: Number of retries on API failure

    Returns:
        List of extracted fact strings
    """
    if not turn_text.strip():
        return []

    client = _get_client()

    user_content = f"Speaker: {speaker}\nDialogue turn: {turn_text}" if speaker else turn_text

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON array from response
            # Handle cases where the model wraps in markdown code blocks
            if "```" in content:
                import re
                json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1).strip()

            facts = json.loads(content)
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
            return []

        except json.JSONDecodeError:
            # Try to salvage partial JSON
            try:
                # Sometimes the model returns facts without proper JSON
                if content.startswith("["):
                    # Find the last complete item
                    content = content.rsplit("]", 1)[0] + "]"
                    facts = json.loads(content)
                    return [str(f) for f in facts if f]
            except (json.JSONDecodeError, IndexError):
                pass
            if attempt < max_retries - 1:
                time.sleep(1)
            continue

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"Warning: LLM extraction failed after {max_retries} retries: {e}")
            return []

    return []


def llm_extract_facts_batch(
    turns: List[dict],
    model: str = "gpt-5-nano",
) -> List[List[str]]:
    """
    Extract facts from multiple dialogue turns.

    Args:
        turns: List of {"speaker": str, "text": str} dicts
        model: OpenAI model to use

    Returns:
        List of fact lists, one per turn
    """
    results = []
    for turn in turns:
        facts = llm_extract_facts(
            turn_text=turn.get("text", ""),
            speaker=turn.get("speaker", ""),
            model=model,
        )
        results.append(facts)
    return results


def extract_facts_from_turn_llm(
    speaker: str,
    text: str,
    model: str = "gpt-5-nano",
) -> List[str]:
    """
    Convenience wrapper matching the signature expected by build_training_data.

    Args:
        speaker: Speaker name
        text: Turn text
        model: LLM model name

    Returns:
        List of extracted facts
    """
    return llm_extract_facts(
        turn_text=text,
        speaker=speaker,
        model=model,
    )
