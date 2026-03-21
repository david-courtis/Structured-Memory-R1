"""Tests for Memory-R1 reward functions."""
import pytest
from memory_r1.reward.em_reward import (
    normalize_answer,
    em_check,
    subem_check,
    extract_answer_from_output,
    compute_score_answer_agent,
)
from memory_r1.agents.memory_manager import (
    compute_memory_manager_reward,
    compute_score_memory_r1,
)


class TestNormalizeAnswer:

    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Answer is: 42!") == "answer is 42"


class TestEMCheck:

    def test_exact_match(self):
        assert em_check("Paris", ["Paris"]) == 1

    def test_case_insensitive(self):
        assert em_check("paris", ["Paris"]) == 1

    def test_no_match(self):
        assert em_check("London", ["Paris"]) == 0

    def test_multiple_answers(self):
        assert em_check("NYC", ["New York City", "NYC", "New York"]) == 1

    def test_articles_ignored(self):
        assert em_check("The Beatles", ["Beatles"]) == 1

    def test_string_input(self):
        assert em_check("Paris", "Paris") == 1


class TestSubEMCheck:

    def test_substring_match(self):
        assert subem_check("The answer is Paris, France", ["Paris"]) == 1

    def test_no_substring(self):
        assert subem_check("London is great", ["Paris"]) == 0


class TestExtractAnswer:

    def test_answer_marker(self):
        output = "**Memories selected:**\n- Memory 1\n**Answer:** Paris"
        assert extract_answer_from_output(output) == "Paris"

    def test_answer_tags(self):
        output = "<answer>42</answer>"
        assert extract_answer_from_output(output) == "42"

    def test_multiple_answers_takes_last(self):
        output = "**Answer:** wrong\n**Answer:** correct"
        assert extract_answer_from_output(output) == "correct"

    def test_no_answer(self):
        output = "I don't know the answer"
        result = extract_answer_from_output(output)
        assert result  # Falls back to last line

    def test_answer_with_newline(self):
        output = "**Answer:** beach\n\nSome extra text"
        assert extract_answer_from_output(output) == "beach"


class TestComputeScoreAnswerAgent:

    def test_correct_answer(self):
        score = compute_score_answer_agent(
            "**Answer:** Paris",
            {"target": ["Paris"]},
        )
        assert score == 1.0

    def test_wrong_answer(self):
        score = compute_score_answer_agent(
            "**Answer:** London",
            {"target": ["Paris"]},
        )
        assert score == 0.0

    def test_no_answer(self):
        score = compute_score_answer_agent(
            "",
            {"target": ["Paris"]},
        )
        assert score == 0.0


class TestComputeScoreMemoryR1:

    def test_correct(self):
        score = compute_score_memory_r1(
            "**Answer:** 2 dogs",
            {"target": ["2 dogs", "two dogs"]},
        )
        assert score == 1.0

    def test_wrong(self):
        score = compute_score_memory_r1(
            "**Answer:** 1 dog",
            {"target": ["2 dogs"]},
        )
        assert score == 0.0

    def test_dict_targets(self):
        """Test with Memory Manager format (QA pair dicts)."""
        score = compute_score_memory_r1(
            "**Answer:** Paris",
            {"target": [{"question": "What city?", "answer": "Paris"}]},
        )
        assert score == 1.0


class TestComputeMemoryManagerReward:

    def test_valid_format(self):
        output = '{"memory": [{"id": "0", "text": "Fact", "event": "ADD"}]}'
        score = compute_memory_manager_reward(
            old_memory=[],
            manager_output=output,
            qa_pairs=[],
        )
        assert score == 0.1  # Format reward

    def test_invalid_format(self):
        score = compute_memory_manager_reward(
            old_memory=[],
            manager_output="This is not JSON",
            qa_pairs=[],
        )
        assert score == 0.0

    def test_with_answer_function(self):
        """Test with a mock answer function."""
        def mock_answer_fn(question, memories):
            # Simple mock: if memories contain "Buddy", answer "Buddy"
            for m in memories:
                if "Buddy" in m.get("text", ""):
                    return "Buddy"
            return "unknown"

        output = '{"memory": [{"id": "0", "text": "Dog named Buddy", "event": "ADD"}]}'
        score = compute_memory_manager_reward(
            old_memory=[],
            manager_output=output,
            qa_pairs=[{"question": "What is the dog's name?", "answer": "Buddy"}],
            answer_fn=mock_answer_fn,
        )
        assert score == 1.0

    def test_with_answer_function_wrong(self):
        def mock_answer_fn(question, memories):
            return "unknown"

        output = '{"memory": [{"id": "0", "text": "Cat named Whiskers", "event": "ADD"}]}'
        score = compute_memory_manager_reward(
            old_memory=[],
            manager_output=output,
            qa_pairs=[{"question": "What is the dog's name?", "answer": "Buddy"}],
            answer_fn=mock_answer_fn,
        )
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
