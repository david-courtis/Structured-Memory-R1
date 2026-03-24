"""
Reward functions for Memory-R1.

Both the Memory Manager and Answer Agent use Exact Match (EM) as the reward
signal (Paper Section 3.1, Eq. 4). The EM score compares the predicted answer
against the gold answer after normalization.

For the Memory Manager, the reward is computed by running the frozen Answer
Agent on the updated memory bank and checking EM of its output.

For the Answer Agent, the reward is the direct EM of its generated answer.
"""
import re
import string
from typing import List, Union


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for exact match comparison.
    Same normalization as Search-R1's qa_em.py.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: Union[str, List[str]]) -> int:
    """
    Check exact match between prediction and any of the golden answers.
    Returns 1 if match, 0 otherwise.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)

    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def subem_check(prediction: str, golden_answers: Union[str, List[str]]) -> int:
    """
    Check if gold answer is a substring of the prediction (after normalization).
    More lenient than strict EM.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)

    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1
    return 0


def extract_answer_from_output(output_text: str) -> str:
    """
    Extract the answer from Answer Agent output.

    The Answer Agent outputs after **Answer:** marker.
    """
    # Try **Answer:** pattern first (also matches **Answer: text** variant)
    pattern = r'\*\*Answer:\*\*\s*(.*?)(?:\n|$)'
    matches = re.findall(pattern, output_text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try **Answer: text** pattern (3B model variant — answer inside bold markers)
    pattern = r'\*\*Answer:\s*(.*?)\*\*'
    matches = re.findall(pattern, output_text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Fallback: try <answer> tags (Search-R1 style)
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, output_text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Last fallback: return last non-empty line
    lines = [l.strip() for l in output_text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return ""


def compute_score_answer_agent(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 0.0,
) -> float:
    """
    Compute EM reward for the Answer Agent.

    Args:
        solution_str: Full decoded sequence (prompt + response)
        ground_truth: Dict with 'target' key containing gold answer(s)
        format_score: Score for valid format but wrong answer

    Returns:
        1.0 for exact match, format_score for wrong answer, 0.0 for no answer
    """
    answer = extract_answer_from_output(solution_str)
    if not answer:
        return 0.0

    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]

    if em_check(answer, targets):
        return 1.0
    elif subem_check(answer, targets):
        return format_score  # Partial credit possible
    return format_score


def compute_score_memory_manager(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 0.0,
) -> float:
    """
    Compute reward for the Memory Manager.

    In the full pipeline, this would:
    1. Parse the manager's output into memory operations
    2. Apply operations to the memory bank
    3. Run the frozen Answer Agent on the updated bank
    4. Return the EM score of the Answer Agent

    For now, we check if the output is valid JSON with memory operations.
    The full frozen-agent reward loop is implemented in the training script.
    """
    from memory_r1.memory_bank import parse_memory_manager_output
    operations, success = parse_memory_manager_output(solution_str)

    if not success:
        return 0.0  # Invalid format

    # Check that operations have valid event types
    valid_events = {"ADD", "UPDATE", "DELETE", "NONE", "NOOP"}
    for op in operations:
        event = op.get("event", "").upper()
        if event not in valid_events:
            return format_score

    # Format is valid - actual reward comes from downstream Answer Agent
    # During training, this is replaced by the frozen-agent EM score
    return format_score
