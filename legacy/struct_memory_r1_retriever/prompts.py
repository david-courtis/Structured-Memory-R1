"""
Prompts for fixed-structure retrieval training.

This package keeps the structured memory fixed and trains a retriever agent
that plans a tree search over the memory schema. The Answer Agent stays frozen.
"""
import json


RETRIEVER_AGENT_SYSTEM = """You are a retrieval agent over a structured memory tree.

The memory tree is fixed. Do not edit memory.
Your job is to plan a search over the tree and return only a JSON object.

Search policy:
- Start from the root and pick relevant branches level by level.
- Use the tree schema and node summaries to decide which branches to expand.
- Use as few branches as possible while keeping enough evidence to answer.
- Prefer concrete fact leaves that directly support the answer.

Output requirements:
- Return ONLY one JSON object.
- Do NOT explain your reasoning.
- Do NOT output markdown fences.
- The JSON must use this schema:
  {
    "levels": [
      {"level": 0, "keys": ["speaker_key"]},
      {"level": 1, "keys": ["topic_key"]},
      {"level": 2, "keys": ["subtopic_or_entity_key"]}
    ],
    "selected_ids": ["memory_id_1", "memory_id_2"],
    "stop": true
  }
- "levels" may be empty if you choose direct memory IDs only.
- "selected_ids" may be empty if branch expansion alone should determine retrieval.
- If nothing is relevant, return {"levels": [], "selected_ids": [], "stop": true}.
"""


ANSWER_AGENT_SYSTEM = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"), calculate the actual date based on the memory timestamp
6. Always convert relative time references to specific dates, months, or years
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words
9. IMPORTANT: Select memories you found that are useful for answering the questions, and output it before you answer questions
10. IMPORTANT: Output the final answer after **Answer:**
"""


def format_answer_agent_prompt(
    question: str,
    memories_speaker_a: list,
    memories_speaker_b: list,
    speaker_a_name: str = "Speaker A",
    speaker_b_name: str = "Speaker B",
) -> str:
    prompt = f"Memories for user {speaker_a_name}:\n"
    for mem in memories_speaker_a:
        prompt += f"- {mem}\n"
    prompt += f"... (In total {len(memories_speaker_a)} most relevant memories from {speaker_a_name}'s Memory Bank are provided) ...\n"

    prompt += f"\nMemories for user {speaker_b_name}:\n"
    for mem in memories_speaker_b:
        prompt += f"- {mem}\n"
    prompt += f"... (In total {len(memories_speaker_b)} most relevant memories from {speaker_b_name}'s Memory Bank are provided) ...\n"

    prompt += f"\nQuestion: {question}"
    return prompt


def make_retriever_training_prompt(
    question: str,
    schema_summary: dict,
    speaker_a_name: str = "Speaker A",
    speaker_b_name: str = "Speaker B",
) -> str:
    schema_text = json.dumps(schema_summary, indent=2)
    return f"""{RETRIEVER_AGENT_SYSTEM}

Conversation participants:
- {speaker_a_name}
- {speaker_b_name}

Question:
{question}

Structured memory schema:
{schema_text}

Return ONLY the retrieval plan as JSON:"""


def make_answer_agent_training_prompt(
    question: str,
    memories_speaker_a: list,
    memories_speaker_b: list,
    speaker_a_name: str = "Speaker A",
    speaker_b_name: str = "Speaker B",
) -> str:
    memories_section = format_answer_agent_prompt(
        question=question,
        memories_speaker_a=memories_speaker_a,
        memories_speaker_b=memories_speaker_b,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
    )

    return f"""{ANSWER_AGENT_SYSTEM}

{memories_section}

Output the memories you selected as relevant using **Memories selected as relevant:** and then provide your final answer after **Answer:**"""
