"""
Functional smoke test for struct_memory_r1_retriever.
"""
from struct_memory_r1_retriever.agents.retriever_agent import (
    FrozenAnswerAgent,
    compute_score_struct_retriever_verl,
    set_frozen_answer_agent,
)
from struct_memory_r1_retriever.data.build_training_data import build_retriever_agent_data
from struct_memory_r1_retriever.data.locomo_loader import Conversation, DialogueTurn, QAPair, Session
from struct_memory_r1_retriever.memory_bank import MemoryBank


class MockAnswerAgent(FrozenAnswerAgent):
    def answer(self, question: str, memories):
        text = " ".join(item.get("text", "") for item in memories)
        question_lower = question.lower()
        if "breed" in question_lower and "golden retriever" in text.lower():
            return "**Answer:** golden retriever"
        if "how many dogs" in question_lower and "two dogs" in text.lower():
            return "**Answer:** 2"
        if "first dog's name" in question_lower and "buddy" in text.lower():
            return "**Answer:** Buddy"
        return "**Answer:** unknown"


def make_conversation() -> Conversation:
    return Conversation(
        sample_id="retriever_smoke_1",
        speaker_a="Alice",
        speaker_b="Bob",
        sessions=[
            Session(
                session_id=1,
                datetime="2023-01-15 10:00",
                turns=[
                    DialogueTurn(speaker="Alice", dia_id="D1:1", text="I just adopted a dog named Buddy from the shelter."),
                    DialogueTurn(speaker="Bob", dia_id="D1:2", text="What breed is he?"),
                    DialogueTurn(speaker="Alice", dia_id="D1:3", text="He's a golden retriever, about 2 years old."),
                ],
            ),
            Session(
                session_id=2,
                datetime="2023-03-20 14:00",
                turns=[
                    DialogueTurn(speaker="Alice", dia_id="D2:1", text="I adopted another dog named Scout."),
                    DialogueTurn(speaker="Alice", dia_id="D2:2", text="I now have two dogs: Buddy and Scout."),
                ],
            ),
        ],
        qa_pairs=[
            QAPair(question="What breed is Buddy?", answer="golden retriever", category="single-hop", evidence=["D1:3"]),
            QAPair(question="How many dogs does Alice have?", answer="2", category="multi-hop", evidence=["D2:2"]),
            QAPair(question="What is Alice's first dog's name?", answer="Buddy", category="single-hop", evidence=["D1:1"]),
        ],
    )


def main():
    conv = make_conversation()
    samples = build_retriever_agent_data([conv], split="train")
    set_frozen_answer_agent(MockAnswerAgent())

    bank = MemoryBank.from_list(__import__("json").loads(samples[0]["extra_info"]["old_memory"]))
    schema = bank.schema_summary()

    plan = {
        "levels": [
            {"level": 0, "keys": ["Alice"]},
            {"level": 1, "keys": ["adopted"]},
        ],
        "selected_ids": [],
        "stop": True,
    }
    score = compute_score_struct_retriever_verl(
        solution_str=__import__("json").dumps(plan),
        ground_truth=samples[2]["reward_model"]["ground_truth"],
        extra_info=samples[2]["extra_info"],
    )
    hits = bank.execute_retrieval_plan(plan, topk=5)

    print("functional_retriever_smoke=PASS")
    print(f"schema_speakers={len(schema.get('speakers', []))}")
    print(f"retriever_samples={len(samples)}")
    print(f"retrieval_hits={len(hits)}")
    print(f"reward_score={score}")


if __name__ == "__main__":
    main()
