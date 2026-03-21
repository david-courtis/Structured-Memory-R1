"""
Memory Manager agent for integration with veRL.

The Memory Manager is a single-turn agent:
1. Receives: current dialogue turn + old memory bank + retrieved facts
2. Outputs: JSON with memory operations {ADD, UPDATE, DELETE, NOOP}
3. Reward: EM score of frozen Answer Agent on the updated memory bank (Eq. 4)

Per the paper (Section 3.1, Algorithm 5), the Memory Manager's reward is
outcome-driven: its operations are judged by their effect on downstream QA.
After applying operations, the updated memory bank is passed to a frozen
Answer Agent, and reward = EM(y_pred, y_gold).
"""
import re
import json
from typing import List, Dict, Tuple, Optional, Callable

from memory_r1.memory_bank import MemoryBank, parse_memory_manager_output
from memory_r1.reward.em_reward import (
    extract_answer_from_output,
    em_check,
    normalize_answer,
)


def postprocess_memory_manager_output(output: str) -> Tuple[Optional[List[dict]], bool]:
    """
    Parse the Memory Manager's generated text into operations.

    Returns:
        (operations, valid_format): list of operations and whether format is valid
    """
    operations, success = parse_memory_manager_output(output)
    return operations, success


def apply_memory_operations(
    old_bank: MemoryBank,
    operations: List[dict],
) -> MemoryBank:
    """
    Apply parsed memory operations to produce a new memory bank.

    Following Algorithm 3 (p.19):
    - ADD: M = M + {f_i}
    - UPDATE: M_tmp = Merge(M_old, f_i); M = (M - M_old) + M_tmp
    - DELETE: M = M - M_ret
    - NOOP: M = M (no change)
    """
    return old_bank.apply_operations(operations)


# =============================================================================
# Frozen Answer Agent wrapper for Memory Manager reward
# =============================================================================

class FrozenAnswerAgent:
    """
    Wrapper for calling a frozen Answer Agent model to compute
    Memory Manager reward (Eq. 4, Algorithm 5).

    The frozen Answer Agent can be:
    1. A local vLLM/HuggingFace model
    2. An API endpoint (e.g., the same model served via vLLM)

    This is used during Memory Manager training to evaluate how well
    the Memory Manager's operations help the Answer Agent answer correctly.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        api_url: Optional[str] = None,
        tokenizer=None,
        model=None,
    ):
        """
        Initialize the frozen Answer Agent.

        Args:
            model_path: Path to a local HuggingFace model
            api_url: URL of a vLLM/API endpoint for the Answer Agent
            tokenizer: Pre-loaded tokenizer (for in-process inference)
            model: Pre-loaded model (for in-process inference)
        """
        self.model_path = model_path
        self.api_url = api_url
        self._tokenizer = tokenizer
        self._model = model

    def answer(self, question: str, memories: List[dict]) -> str:
        """
        Generate an answer given a question and memory bank.

        This is the frozen Answer Agent inference call used in
        Algorithm 5 (line 23): r_i ~ L_a(p_i)

        Args:
            question: The question to answer
            memories: List of {id, text} memory entries

        Returns:
            The predicted answer string
        """
        from memory_r1.prompts import ANSWER_AGENT_SYSTEM

        # Build the prompt
        memories_text = "\n".join(
            f"- {m.get('text', '')}" for m in memories
        )
        prompt = f"""{ANSWER_AGENT_SYSTEM}

Retrieved Memories:
{memories_text}

Question: {question}

Output the memories you selected as relevant using **Memories selected as relevant:** and then provide your final answer after **Answer:**"""

        # Try API endpoint first
        if self.api_url:
            return self._answer_via_api(prompt)

        # Fall back to local model
        if self._model is not None and self._tokenizer is not None:
            return self._answer_via_model(prompt)

        # No model available — return empty
        return ""

    def _answer_via_api(self, prompt: str) -> str:
        """Call the Answer Agent via an API endpoint."""
        import requests
        try:
            response = requests.post(
                self.api_url,
                json={
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.0,  # Greedy for evaluation (paper Section D)
                },
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                generated = result.get("text", result.get("choices", [{}])[0].get("text", ""))
                return extract_answer_from_output(generated) or ""
        except Exception as e:
            print(f"Warning: Answer Agent API call failed: {e}")
        return ""

    def _answer_via_model(self, prompt: str) -> str:
        """Call the Answer Agent via a local model."""
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with __import__("torch").no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )
        generated = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_answer_from_output(generated) or ""


# Global frozen Answer Agent instance (set during training setup)
_frozen_answer_agent: Optional[FrozenAnswerAgent] = None


def set_frozen_answer_agent(agent: FrozenAnswerAgent):
    """Set the global frozen Answer Agent for Memory Manager reward computation."""
    global _frozen_answer_agent
    _frozen_answer_agent = agent


def get_frozen_answer_agent() -> Optional[FrozenAnswerAgent]:
    """Get the global frozen Answer Agent."""
    return _frozen_answer_agent


# =============================================================================
# Reward computation
# =============================================================================

def compute_memory_manager_reward(
    old_memory: List[dict],
    manager_output: str,
    qa_pairs: List[dict],
    answer_fn: Optional[Callable] = None,
) -> float:
    """
    Compute reward for the Memory Manager (Eq. 4, Algorithm 5).

    Full pipeline:
    1. Parse manager output into memory operations
    2. Apply operations to old memory bank → new memory bank
    3. For each QA pair, run the frozen Answer Agent on updated bank
    4. R = mean EM(y_pred, y_gold)

    When no answer function is available, falls back to format reward.

    Args:
        old_memory: List of {id, text} dicts (current memory state)
        manager_output: Raw text output from Memory Manager
        qa_pairs: List of {question, answer} dicts for reward evaluation
        answer_fn: Optional callable(question, memories) -> answer_text
                   If None, uses the global frozen Answer Agent

    Returns:
        Reward score in [0, 1]
    """
    # Step 1: Parse operations
    operations, valid = postprocess_memory_manager_output(manager_output)
    if not valid:
        return 0.0

    # Step 2: Apply operations to memory bank
    old_bank = MemoryBank.from_list(old_memory) if old_memory else MemoryBank()
    try:
        new_bank = apply_memory_operations(old_bank, operations)
    except Exception:
        return 0.0

    # Step 3: Resolve answer function
    if answer_fn is None:
        agent = get_frozen_answer_agent()
        if agent is not None:
            answer_fn = agent.answer

    # Step 4: Compute EM reward via frozen Answer Agent
    if answer_fn is not None and qa_pairs:
        new_memories = new_bank.to_list()
        total_score = 0.0
        valid_pairs = 0
        for qa in qa_pairs:
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "")
            if not question or not gold_answer:
                continue

            # Run frozen Answer Agent (Algorithm 5, line 23)
            predicted_answer = answer_fn(question, new_memories)
            predicted_answer = extract_answer_from_output(predicted_answer) if predicted_answer else ""
            if em_check(predicted_answer, [gold_answer]):
                total_score += 1.0
            valid_pairs += 1

        if valid_pairs > 0:
            return total_score / valid_pairs
        return 0.1  # Format correct but no valid QA pairs

    # Without answer function, reward valid format (small reward)
    return 0.1


def extract_solution_memory_r1(solution_str: str) -> Optional[str]:
    """
    Extract the answer from a Memory-R1 Answer Agent response.

    Looks for:
    1. **Answer:** marker (Memory-R1 style)
    2. <answer>...</answer> tags (Search-R1 style)
    """
    return extract_answer_from_output(solution_str)


def compute_score_memory_r1(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 0.0,
) -> float:
    """
    Compute EM score for Memory-R1 Answer Agent output.
    Compatible with veRL's reward function interface.

    Args:
        solution_str: Full decoded sequence
        ground_truth: Dict with 'target' key
        format_score: Score for valid format but wrong answer

    Returns:
        1.0 for EM, format_score for wrong answer, 0.0 for no answer
    """
    answer = extract_solution_memory_r1(solution_str)

    if answer is None or answer == "":
        return 0.0

    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]

    # Handle list of dicts (Memory Manager format with QA pairs)
    if targets and isinstance(targets[0], dict):
        targets = [t.get("answer", "") for t in targets if t.get("answer")]

    if not targets:
        return 0.0

    if em_check(answer, targets):
        return 1.0
    return format_score


def compute_score_memory_manager_verl(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 0.0,
    extra_info: Optional[dict] = None,
) -> float:
    """
    Compute reward for Memory Manager output in veRL's reward function interface.

    This is the reward function called from _select_rm_score_fn in main_ppo.py.
    It parses the Memory Manager's output, applies operations, and evaluates
    via the frozen Answer Agent if available.

    The ground_truth dict contains:
    - target: List of {question, answer} QA pair dicts

    The extra_info dict (passed from RewardManager) contains:
    - old_memory: JSON string of the memory bank state before this turn
    - facts: JSON string of the extracted facts for this turn

    Args:
        solution_str: Full decoded sequence (prompt + response)
        ground_truth: Dict with 'target' key containing QA pairs
        format_score: Score for valid format but wrong answer
        extra_info: Dict with old_memory and facts from the parquet data

    Returns:
        Reward score in [0, 1]
    """
    from memory_r1.memory_bank import parse_memory_manager_output

    # Extract just the response part (after the prompt)
    # The Memory Manager's output is JSON with memory operations
    operations, valid = parse_memory_manager_output(solution_str)

    if not valid:
        return 0.0

    # Check valid event types
    valid_events = {"ADD", "UPDATE", "DELETE", "NONE", "NOOP"}
    for op in operations:
        event = op.get("event", "").upper()
        if event not in valid_events:
            return 0.0

    # Try to compute full reward with frozen Answer Agent
    agent = get_frozen_answer_agent()
    if agent is not None:
        targets = ground_truth.get("target", [])
        if isinstance(targets, list) and targets and isinstance(targets[0], dict):
            qa_pairs = targets

            # Reconstruct old memory bank from extra_info (Algorithm 5)
            old_bank = MemoryBank()
            if extra_info:
                old_memory_str = extra_info.get("old_memory", "[]")
                try:
                    old_memory_list = json.loads(old_memory_str) if isinstance(old_memory_str, str) else old_memory_str
                    if old_memory_list:
                        old_bank = MemoryBank.from_list(old_memory_list)
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass  # Fall back to empty bank

            try:
                new_bank = old_bank.apply_operations(operations)
            except Exception:
                return format_score

            new_memories = new_bank.to_list()
            total_score = 0.0
            valid_pairs = 0
            for qa in qa_pairs:
                question = qa.get("question", "")
                gold_answer = qa.get("answer", "")
                if not question or not gold_answer:
                    continue
                predicted = agent.answer(question, new_memories)
                predicted = extract_answer_from_output(predicted) if predicted else ""
                if em_check(predicted, [gold_answer]):
                    total_score += 1.0
                valid_pairs += 1

            if valid_pairs > 0:
                return total_score / valid_pairs

    # Format-only reward when no frozen Answer Agent is available
    return format_score
