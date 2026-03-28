"""
Retriever agent for fixed structured memory.

The structured memory tree is fixed. The trainable policy emits a search plan
that selects branches and/or leaves from the tree, then a frozen Answer Agent
answers using the retrieved evidence.
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from struct_memory_r1_retriever.memory_bank import MemoryBank
from struct_memory_r1_retriever.reward.em_reward import (
    extract_answer_from_output,
    em_check,
    token_f1,
)


def parse_retriever_output(output_text: str) -> Tuple[Dict[str, Any], bool]:
    """Parse the first JSON object from the model output."""
    if not output_text:
        return {}, False
    match = re.search(r"\{[\s\S]*\}", output_text)
    if match is None:
        return {}, False
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    payload.setdefault("levels", [])
    payload.setdefault("selected_ids", [])
    payload.setdefault("stop", True)
    if not isinstance(payload["levels"], list) or not isinstance(payload["selected_ids"], list):
        return {}, False
    return payload, True


class FrozenAnswerAgent:
    def __init__(
        self,
        model_path: Optional[str] = None,
        api_url: Optional[str] = None,
        tokenizer=None,
        model=None,
    ):
        self.model_path = model_path
        self.api_url = api_url
        self._tokenizer = tokenizer
        self._model = model

    def answer(self, question: str, memories: List[dict]) -> str:
        from struct_memory_r1_retriever.prompts import ANSWER_AGENT_SYSTEM

        memories_text = "\n".join(f"- {m.get('text', '')}" for m in memories)
        prompt = f"""{ANSWER_AGENT_SYSTEM}

Retrieved Memories:
{memories_text}

Question: {question}

Output the memories you selected as relevant using **Memories selected as relevant:** and then provide your final answer after **Answer:**"""

        if self.api_url:
            return self._answer_via_api(prompt)
        if self._model is not None and self._tokenizer is not None:
            return self._answer_via_model(prompt)
        return ""

    def _answer_via_api(self, prompt: str) -> str:
        import requests
        try:
            response = requests.post(
                self.api_url,
                json={"prompt": prompt, "max_tokens": 256, "temperature": 0.0},
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                generated = result.get("text", result.get("choices", [{}])[0].get("text", ""))
                return extract_answer_from_output(generated) or ""
        except Exception:
            return ""
        return ""

    def _answer_via_model(self, prompt: str) -> str:
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


_frozen_answer_agent: Optional[FrozenAnswerAgent] = None


def set_frozen_answer_agent(agent: FrozenAnswerAgent):
    global _frozen_answer_agent
    _frozen_answer_agent = agent


def get_frozen_answer_agent() -> Optional[FrozenAnswerAgent]:
    return _frozen_answer_agent


def _coerce_targets_to_list(targets: Any) -> List[Any]:
    if targets is None:
        return []
    if isinstance(targets, list):
        return targets
    if hasattr(targets, "tolist"):
        try:
            converted = targets.tolist()
            return converted if isinstance(converted, list) else [converted]
        except Exception:
            return []
    return [targets]


def _evidence_bonus(selected: List[dict], gold_ids: List[str]) -> float:
    if not gold_ids:
        return 0.0
    selected_ids = {str(item.get("id", "")) for item in selected}
    hits = len(selected_ids & {str(gid) for gid in gold_ids})
    if hits == 0:
        return 0.0
    return min(0.2, 0.1 * hits)


def compute_score_struct_retriever_verl(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 0.0,
    extra_info: Optional[dict] = None,
) -> float:
    """
    Reward for the retriever agent.

    The retriever policy emits a search plan. The fixed structured memory bank
    executes that plan, and a frozen Answer Agent answers from the retrieved
    leaves. Reward is answer correctness plus a small evidence-hit bonus.
    """
    plan, valid = parse_retriever_output(solution_str)
    if not valid:
        return 0.0

    old_bank = MemoryBank()
    question = ""
    gold_answers: List[str] = []
    gold_entry_ids: List[str] = []
    if extra_info:
        old_memory = extra_info.get("old_memory", "[]")
        try:
            old_memory_list = json.loads(old_memory) if isinstance(old_memory, str) else old_memory
            if old_memory_list:
                old_bank = MemoryBank.from_list(old_memory_list)
        except (TypeError, json.JSONDecodeError):
            pass
        question = str(extra_info.get("question", ""))
        raw_gold_ids = extra_info.get("gold_entry_ids", [])
        if isinstance(raw_gold_ids, str):
            try:
                raw_gold_ids = json.loads(raw_gold_ids)
            except json.JSONDecodeError:
                raw_gold_ids = [raw_gold_ids]
        gold_entry_ids = [str(x) for x in _coerce_targets_to_list(raw_gold_ids)]

    targets = _coerce_targets_to_list(ground_truth.get("target", []))
    if targets and isinstance(targets[0], dict):
        question = question or str(targets[0].get("question", ""))
        gold_answers = [str(item.get("answer", "")) for item in targets if isinstance(item, dict) and item.get("answer")]
    else:
        gold_answers = [str(item) for item in targets if str(item)]

    retrieved = old_bank.execute_retrieval_plan(plan, topk=30)
    evidence_score = _evidence_bonus(retrieved, gold_entry_ids)

    agent = get_frozen_answer_agent()
    format_bonus = 0.1
    if agent is None or not question or not gold_answers:
        return min(1.0, format_score + format_bonus + evidence_score)

    predicted = agent.answer(question, retrieved)
    predicted = extract_answer_from_output(predicted) if predicted else ""
    if em_check(predicted, gold_answers):
        return min(1.0, 0.8 + format_bonus + evidence_score)
    f1 = token_f1(predicted, gold_answers)
    if f1 > 0.0:
        return min(1.0, f1 + format_bonus + evidence_score)
    return min(1.0, format_score + format_bonus + evidence_score)
