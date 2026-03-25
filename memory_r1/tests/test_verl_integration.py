"""Tests for veRL integration: reward routing, data format, and frozen agent."""
import pytest
import os


class TestRewardRouting:
    """Test that _select_rm_score_fn routes correctly for Memory-R1 data sources."""

    def test_answer_agent_routing(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("answer_agent")
        assert callable(fn)

    def test_memory_manager_routing(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("memory_manager")
        assert callable(fn)

    def test_nq_still_works(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("nq")
        assert callable(fn)

    def test_unknown_raises(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        with pytest.raises(NotImplementedError):
            _select_rm_score_fn("unknown_source")


class TestFrozenCheckpointResolution:
    """Test path resolution for frozen Answer Agent checkpoints."""

    def test_accepts_direct_hf_checkpoint_dir(self, tmp_path):
        from verl.trainer.main_ppo import _resolve_frozen_answer_agent_path

        ckpt = tmp_path / "global_step_100"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}", encoding="utf-8")
        (ckpt / "model.safetensors").write_text("", encoding="utf-8")

        assert _resolve_frozen_answer_agent_path(str(ckpt)) == str(ckpt.resolve())

    def test_resolves_latest_actor_global_step_from_run_root(self, tmp_path):
        from verl.trainer.main_ppo import _resolve_frozen_answer_agent_path

        actor_dir = tmp_path / "actor"
        step_100 = actor_dir / "global_step_100"
        step_200 = actor_dir / "global_step_200"
        step_100.mkdir(parents=True)
        step_200.mkdir(parents=True)
        (step_100 / "config.json").write_text("{}", encoding="utf-8")
        (step_100 / "model.safetensors").write_text("", encoding="utf-8")
        (step_200 / "config.json").write_text("{}", encoding="utf-8")
        (step_200 / "model.safetensors").write_text("", encoding="utf-8")

        assert _resolve_frozen_answer_agent_path(str(tmp_path)) == str(step_200.resolve())

    def test_rejects_run_root_without_hf_export(self, tmp_path):
        from verl.trainer.main_ppo import _resolve_frozen_answer_agent_path

        actor_dir = tmp_path / "actor"
        step_100 = actor_dir / "global_step_100"
        step_100.mkdir(parents=True)
        (step_100 / "config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(OSError, match="No Hugging Face checkpoint found"):
            _resolve_frozen_answer_agent_path(str(tmp_path))

    def test_prefers_best_checkpoint_when_present(self, tmp_path):
        from verl.trainer.main_ppo import _resolve_frozen_answer_agent_path

        best_dir = tmp_path / "best"
        actor_step = tmp_path / "actor" / "global_step_100"
        best_dir.mkdir(parents=True)
        actor_step.mkdir(parents=True)
        (best_dir / "config.json").write_text("{}", encoding="utf-8")
        (best_dir / "model.safetensors").write_text("", encoding="utf-8")
        (actor_step / "config.json").write_text("{}", encoding="utf-8")
        (actor_step / "model.safetensors").write_text("", encoding="utf-8")

        assert _resolve_frozen_answer_agent_path(str(tmp_path)) == str(best_dir.resolve())


class TestBestCheckpointMetricSelection:
    """Test how the trainer chooses the validation metric for best checkpointing."""

    def test_prefers_single_validation_score(self):
        from types import SimpleNamespace
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = SimpleNamespace(trainer={})

        metric_name, metric_value = trainer._select_best_validation_metric({
            "val/test_score/memory_manager": 0.72,
        })

        assert metric_name == "val/test_score/memory_manager"
        assert metric_value == 0.72

    def test_prefers_memory_manager_metric_when_multiple_scores_exist(self):
        from types import SimpleNamespace
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = SimpleNamespace(trainer={})

        metric_name, metric_value = trainer._select_best_validation_metric({
            "val/test_score/answer_agent": 0.61,
            "val/test_score/memory_manager": 0.74,
        })

        assert metric_name == "val/test_score/memory_manager"
        assert metric_value == 0.74


class TestAnswerAgentRewardWithData:
    """Test reward function with actual parquet data format."""

    def test_correct_answer(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("answer_agent")

        score = fn(
            solution_str="**Memories selected as relevant:**\n- Memory 1\n**Answer:** Paris",
            ground_truth={"target": ["Paris"]},
        )
        assert score == 1.0

    def test_wrong_answer(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("answer_agent")

        score = fn(
            solution_str="**Answer:** London",
            ground_truth={"target": ["Paris"]},
        )
        assert score == 0.0

    def test_with_dict_targets(self):
        """Memory Manager format: targets are QA pair dicts."""
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("answer_agent")

        score = fn(
            solution_str="**Answer:** Buddy",
            ground_truth={"target": [{"question": "Dog name?", "answer": "Buddy"}]},
        )
        assert score == 1.0


class TestMemoryManagerRewardWithData:
    """Test Memory Manager reward function (compute_score_memory_manager_verl)."""

    def test_valid_json_output(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("memory_manager")

        score = fn(
            solution_str='{"memory": [{"id": "0", "text": "Fact", "event": "ADD"}]}',
            ground_truth={"target": [{"question": "Q?", "answer": "A"}]},
        )
        # Format reward (no frozen Answer Agent available)
        assert score == 0.0  # format_score defaults to 0.0

    def test_invalid_json_output(self):
        from verl.trainer.main_ppo import _select_rm_score_fn
        fn = _select_rm_score_fn("memory_manager")

        score = fn(
            solution_str="This is not valid JSON",
            ground_truth={"target": []},
        )
        assert score == 0.0


class TestFrozenAnswerAgent:
    """Test the frozen Answer Agent wrapper."""

    def test_frozen_agent_set_get(self):
        from memory_r1.agents.memory_manager import (
            FrozenAnswerAgent, set_frozen_answer_agent, get_frozen_answer_agent,
        )
        agent = FrozenAnswerAgent()
        set_frozen_answer_agent(agent)
        assert get_frozen_answer_agent() is agent
        # Clean up
        set_frozen_answer_agent(None)

    def test_reward_with_mock_frozen_agent(self):
        """Test Memory Manager reward with a mock frozen Answer Agent."""
        from memory_r1.agents.memory_manager import (
            FrozenAnswerAgent, set_frozen_answer_agent,
            compute_score_memory_manager_verl,
        )

        class MockAgent(FrozenAnswerAgent):
            def answer(self, question, memories):
                for m in memories:
                    if "Buddy" in m.get("text", ""):
                        return "**Answer:** Buddy"
                return "**Answer:** unknown"

        agent = MockAgent()
        set_frozen_answer_agent(agent)

        try:
            score = compute_score_memory_manager_verl(
                solution_str='{"memory": [{"id": "0", "text": "Dog named Buddy", "event": "ADD"}]}',
                ground_truth={"target": [{"question": "Dog name?", "answer": "Buddy"}]},
            )
            assert score == 1.0

            # Wrong answer
            score = compute_score_memory_manager_verl(
                solution_str='{"memory": [{"id": "0", "text": "Cat named Whiskers", "event": "ADD"}]}',
                ground_truth={"target": [{"question": "Dog name?", "answer": "Buddy"}]},
            )
            assert score == 0.0
        finally:
            set_frozen_answer_agent(None)

    def test_reward_with_extra_info_old_memory(self):
        """Test that extra_info.old_memory is used for UPDATE/DELETE operations."""
        from memory_r1.agents.memory_manager import (
            FrozenAnswerAgent, set_frozen_answer_agent,
            compute_score_memory_manager_verl,
        )
        import json

        class MockAgent(FrozenAnswerAgent):
            def answer(self, question, memories):
                # Check if the updated memory contains both dogs
                for m in memories:
                    if "Buddy" in m.get("text", "") and "Scout" in m.get("text", ""):
                        return "**Answer:** 2"
                return "**Answer:** 1"

        agent = MockAgent()
        set_frozen_answer_agent(agent)

        try:
            # UPDATE operation that merges "adopted Buddy" with "also adopted Scout"
            old_memory = [{"id": "0", "text": "Alice adopted a dog named Buddy"}]
            score = compute_score_memory_manager_verl(
                solution_str='{"memory": [{"id": "0", "text": "Alice adopted dogs Buddy and Scout", "event": "UPDATE", "old_memory": "Alice adopted a dog named Buddy"}]}',
                ground_truth={"target": [{"question": "How many dogs?", "answer": "2"}]},
                extra_info={"old_memory": json.dumps(old_memory)},
            )
            assert score == 1.0

            # Without extra_info, UPDATE on empty bank falls back to ADD (still works for this case)
            score_no_extra = compute_score_memory_manager_verl(
                solution_str='{"memory": [{"id": "0", "text": "Alice adopted dogs Buddy and Scout", "event": "UPDATE", "old_memory": "Alice adopted a dog named Buddy"}]}',
                ground_truth={"target": [{"question": "How many dogs?", "answer": "2"}]},
            )
            # Should still work because UPDATE on missing ID falls back to ADD
            assert score_no_extra == 1.0
        finally:
            set_frozen_answer_agent(None)


class TestParquetDataFormat:
    """Test that saved parquet files have the correct format for veRL."""

    @pytest.fixture
    def aa_parquet(self):
        path = "data/memory_r1/answer_agent/train.parquet"
        if not os.path.exists(path):
            pytest.skip("Run build_training_data first")
        import pandas as pd
        return pd.read_parquet(path)

    @pytest.fixture
    def mm_parquet(self):
        path = "data/memory_r1/memory_manager/train.parquet"
        if not os.path.exists(path):
            pytest.skip("Run build_training_data first")
        import pandas as pd
        return pd.read_parquet(path)

    def test_aa_required_columns(self, aa_parquet):
        required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        assert required.issubset(set(aa_parquet.columns))

    def test_aa_prompt_format(self, aa_parquet):
        row = aa_parquet.iloc[0]
        prompt = row["prompt"]
        assert len(prompt) >= 1
        assert prompt[0]["role"] == "user"
        assert "content" in prompt[0]

    def test_aa_prompt_has_speaker_grouping(self, aa_parquet):
        """Answer Agent prompts should have speaker-grouped memories."""
        row = aa_parquet.iloc[0]
        content = row["prompt"][0]["content"]
        assert "Memories for user" in content

    def test_aa_prompt_has_distillation_instruction(self, aa_parquet):
        """Answer Agent prompts should include Memory Distillation instruction."""
        row = aa_parquet.iloc[0]
        content = row["prompt"][0]["content"]
        assert "Memories selected as relevant" in content

    def test_aa_reward_model_format(self, aa_parquet):
        row = aa_parquet.iloc[0]
        rm = row["reward_model"]
        assert "ground_truth" in rm
        assert "target" in rm["ground_truth"]

    def test_aa_data_source(self, aa_parquet):
        assert all(aa_parquet["data_source"] == "answer_agent")

    def test_mm_required_columns(self, mm_parquet):
        required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        assert required.issubset(set(mm_parquet.columns))

    def test_mm_data_source(self, mm_parquet):
        assert all(mm_parquet["data_source"] == "memory_manager")

    def test_mm_prompt_contains_full_examples(self, mm_parquet):
        """Memory Manager prompt should include paper examples."""
        row = mm_parquet.iloc[0]
        content = row["prompt"][0]["content"]
        assert "User is a software engineer" in content
        assert "cheese pizza" in content

    def test_mm_has_turn_idx(self, mm_parquet):
        """Memory Manager data should have per-turn granularity info."""
        row = mm_parquet.iloc[0]
        assert "turn_idx" in row["extra_info"]

    def test_mm_per_turn_count(self, mm_parquet):
        """Memory Manager should have many more samples than sessions."""
        # With per-turn granularity, we should have >> 2 sessions worth
        assert len(mm_parquet) > 100

    def test_aa_split_matches_paper(self, aa_parquet):
        """Paper reports 152 train + 81 val = 233 total in train parquet."""
        assert len(aa_parquet) == 233


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
