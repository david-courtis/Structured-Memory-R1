#!/bin/bash
# StructMemoryR1 — Stage 2: Retrieve Agent GRPO training (single GPU)
#
# The Retrieve Agent plans a structured search over the memory tree schema.
# It receives: question + tree schema
# It outputs:  JSON search plan {levels, selected_ids, stop}
#
# Reward: r_R = R(y, y*) + lambda_ev * Hit(R, E*) - lambda_cost * C(z)
#         where R is frozen Answer Agent F1, Hit is evidence coverage, C is cost.
#
# Requires: FROZEN_ANSWER_AGENT env var pointing to Stage 1 checkpoint

set -e

PYTHON="conda run --no-capture-output -n searchr1 python3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# Build data if needed
if [ ! -f "${REPO_DIR}/data/struct_memory_r1/retrieve_agent/train.parquet" ]; then
    echo "Building StructMemoryR1 retriever training data from LoCoMo..."
    $PYTHON -m struct_memory_r1.data.build_training_data \
        --output_dir "${REPO_DIR}/data/struct_memory_r1" \
        --stage retrieve_agent
fi

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_LAUNCH_BLOCKING=1

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export EXPERIMENT_NAME="struct-r1-retrieve-agent-grpo-$(echo $BASE_MODEL | tr '/' '-' | tr '[:upper:]' '[:lower:]')"

# Frozen Answer Agent from Stage 1
FROZEN_ANSWER_AGENT="${FROZEN_ANSWER_AGENT:-}"
EXTRA_ARGS=""
if [ -n "$FROZEN_ANSWER_AGENT" ]; then
    echo "Using frozen Answer Agent from: $FROZEN_ANSWER_AGENT"
    EXTRA_ARGS="+frozen_answer_agent_path=$FROZEN_ANSWER_AGENT"
else
    echo "[WARN] No FROZEN_ANSWER_AGENT set. Reward will be format-only."
fi

PYTHONUNBUFFERED=1 $PYTHON -m verl.trainer.main_ppo \
    data.train_files="${REPO_DIR}/data/struct_memory_r1/retrieve_agent/train.parquet" \
    data.val_files="${REPO_DIR}/data/struct_memory_r1/retrieve_agent/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.max_start_length=512 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=false \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=StructMemoryR1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir="${REPO_DIR}/verl_checkpoints/$EXPERIMENT_NAME" \
    do_search=false \
    max_turns=1 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    $EXTRA_ARGS \
    2>&1 | tee "${REPO_DIR}/$EXPERIMENT_NAME.log"
