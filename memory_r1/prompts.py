"""
Prompts for Memory-R1 Memory Manager and Answer Agent.

Faithfully reproduced from the Memory-R1 paper (Yan et al., 2025):
- Memory Manager Prompt: Figures 9-10 (verbatim)
- Answer Agent Prompt: Figure 11 (verbatim)
- LLM-as-a-Judge Prompt: Figure 12 (verbatim)
"""
import json
from typing import List, Optional

# =============================================================================
# Memory Manager Prompt (Paper Figures 9-10, verbatim)
# =============================================================================

MEMORY_MANAGER_SYSTEM = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.

- Example:
    Old Memory:
    [
        {"id" : "0", "text" : "User is a software engineer"}
    ]
    Retrieved facts: ["Name is John"]

    New Memory:
    {
        "memory" : [
            {"id" : "0", "text" : "User is a software engineer", "event" : "NONE"},
            {"id" : "1", "text" : "Name is John", "event" : "ADD"}
        ]
    }

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it.

If the retrieved fact contains information that conveys the same thing as the memory, keep the version with more detail.

Example (a) - if the memory contains "User likes to play cricket" and the retrieved fact is "Loves to play cricket with friends", then update the memory with the retrieved fact.

Example (b) - if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves cheese pizza", then do NOT update it because they convey the same information.

Important: When updating, keep the same ID and preserve old_memory.

- Example:
    Old Memory:
    [
        {"id" : "0", "text" : "I really like cheese pizza"},
        {"id" : "2", "text" : "User likes to play cricket"}
    ]
    Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]

    New Memory:
    {
        "memory" : [
            {"id" : "0", "text" : "Loves cheese and chicken pizza", "event" : "UPDATE",
             "old_memory" : "I really like cheese pizza"},
            {"id" : "2", "text" : "Loves to play cricket with friends", "event" : "UPDATE",
             "old_memory" : "User likes to play cricket"}
        ]
    }

3. **Delete**: If the retrieved facts contain information that contradicts the memory, delete it. When deleting, return the same IDs - do not generate new IDs.

- Example:
    Old Memory:
    [
        {"id" : "1", "text" : "Loves cheese pizza"}
    ]
    Retrieved facts: ["Dislikes cheese pizza"]

    New Memory:
    {
        "memory" : [
            {"id" : "1", "text" : "Loves cheese pizza", "event" : "DELETE"}
        ]
    }

4. **No Change**: If the retrieved facts are already present, make no change.

- Example:
    Old Memory:
    [
        {"id" : "0", "text" : "Name is John"}
    ]
    Retrieved facts: ["Name is John"]

    New Memory:
    {
        "memory" : [
            {"id" : "0", "text" : "Name is John", "event" : "NONE"}
        ]
    }"""


def format_memory_manager_prompt(
    old_memory: list,
    retrieved_facts: list,
) -> str:
    """
    Format the Memory Manager prompt with current memory state and new facts.

    Args:
        old_memory: List of {id, text} dicts representing current memory bank
        retrieved_facts: List of new facts extracted from the dialogue turn

    Returns:
        Formatted prompt string
    """
    memory_str = "Old Memory:\n"
    if old_memory:
        memory_str += json.dumps(old_memory, indent=2)
    else:
        memory_str += "    []"

    facts_str = "Retrieved facts: " + json.dumps(retrieved_facts)

    return f"{memory_str}\n{facts_str}\n\nNew Memory:"


# =============================================================================
# Answer Agent Prompt (Paper Figure 11, verbatim)
# =============================================================================

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

# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions"""


def format_answer_agent_prompt(
    question: str,
    memories_speaker_a: list,
    memories_speaker_b: list,
    speaker_a_name: str = "Speaker A",
    speaker_b_name: str = "Speaker B",
) -> str:
    """
    Format the Answer Agent prompt with retrieved memories grouped by speaker.

    Following Figure 11, memories are presented as:
      Memories for user <Name>:
      - <timestamp>: <memory text>
      ...
      (In total N most relevant memories from <Name>'s Memory Bank are provided)

    Args:
        question: The question to answer
        memories_speaker_a: Retrieved memories for speaker A (strings with timestamps)
        memories_speaker_b: Retrieved memories for speaker B (strings with timestamps)
        speaker_a_name: Name of speaker A
        speaker_b_name: Name of speaker B

    Returns:
        Formatted prompt string
    """
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


# =============================================================================
# Training prompt constructors for veRL
# =============================================================================

def make_memory_manager_training_prompt(
    dialogue_turn: str,
    old_memory: list,
    retrieved_facts: list,
) -> str:
    """
    Create a training prompt for the Memory Manager in veRL format.

    This combines the system prompt with the specific turn context,
    formatted as a single user message for the chat template.
    """
    memory_str = json.dumps(old_memory, indent=2) if old_memory else "[]"
    facts_str = json.dumps(retrieved_facts)

    content = f"""{MEMORY_MANAGER_SYSTEM}

Current dialogue turn:
{dialogue_turn}

Old Memory:
{memory_str}

Retrieved facts: {facts_str}

Output the updated memory as JSON:"""

    return content


def make_answer_agent_training_prompt(
    question: str,
    memories_speaker_a: list,
    memories_speaker_b: list,
    speaker_a_name: str = "Speaker A",
    speaker_b_name: str = "Speaker B",
) -> str:
    """
    Create a training prompt for the Answer Agent in veRL format.

    Following Figure 11, this includes:
    - System instructions
    - Speaker-grouped memories with timestamps
    - Question
    - Memory Distillation output format instruction

    The Answer Agent is expected to output:
    **Memories selected as relevant:**
    - <selected memories>
    **Answer:** <concise answer>
    """
    # Build speaker-grouped memories section
    memories_section = format_answer_agent_prompt(
        question=question,
        memories_speaker_a=memories_speaker_a,
        memories_speaker_b=memories_speaker_b,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
    )

    content = f"""{ANSWER_AGENT_SYSTEM}

{memories_section}

Output the memories you selected as relevant using **Memories selected as relevant:** and then provide your final answer after **Answer:**"""

    return content


# =============================================================================
# LLM-as-a-Judge Prompt (Paper Figure 12, verbatim)
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer,
which you will score as CORRECT or WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time-related questions, the gold answer will be a specific date, month, or year. The generated answer might include relative references (e.g., "last Tuesday"), but you should be generous - if it refers to the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs. "7 May").

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Return the label in JSON format with the key as "label"."""
