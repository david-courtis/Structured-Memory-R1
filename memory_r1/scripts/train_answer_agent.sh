#!/bin/bash
# Memory-R1: Answer Agent GRPO training on a single 24GB GPU
#
# Two-stage training (paper Section 3, p.16-17):
# Stage 1: Train Answer Agent first (this script)
# Stage 2: Train Memory Manager with frozen Answer Agent (run train_memory_manager.sh)
#
# The Answer Agent receives retrieved memories and generates answers.
# Reward: R = EM(y_pred, y_gold) (paper Eq. 5, Section 3.2)

set -e

# Ensure we use the searchr1 conda environment's Python
PYTHON="conda run --no-capture-output -n searchr1 python3"

# Step 1: Build training data if not present
if [ ! -f data/memory_r1/answer_agent/train.parquet ]; then
    echo "Building Memory-R1 training data from LoCoMo..."
    $PYTHON -m memory_r1.data.build_training_data
fi

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
# WSL2 CUDA workaround: synchronous launches prevent "unknown error"
export CUDA_LAUNCH_BLOCKING=1

export BASE_MODEL='Qwen/Qwen2.5-0.5B'
export EXPERIMENT_NAME=memory-r1-answer-agent-grpo-qwen2.5-0.5b

PYTHONUNBUFFERED=1 $PYTHON -m verl.trainer.main_ppo \
    data.train_files=data/memory_r1/answer_agent/train.parquet \
    data.val_files=data/memory_r1/answer_agent/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
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
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.actor.state_masking=false \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=Memory-R1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=750 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    do_search=false \
    max_turns=1 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
