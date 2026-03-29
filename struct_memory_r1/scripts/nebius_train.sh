#!/bin/bash
# =============================================================================
# StructMemoryR1: Full 3-Stage Training Pipeline for Cloud GPU
#
# Usage:
#   bash struct_memory_r1/scripts/nebius_train.sh [--stage <1|2|3|all>] [--model <hf_model>]
#
# Stages:
#   1   - Train Answer Agent only
#   2   - Train Retrieve Agent (requires Stage 1 checkpoint)
#   3   - Train Memory Manager (requires Stage 1 + 2 checkpoints)
#   all - Run all three stages sequentially (default)
#
# Examples:
#   bash struct_memory_r1/scripts/nebius_train.sh
#   bash struct_memory_r1/scripts/nebius_train.sh --stage 1 --model Qwen/Qwen2.5-7B-Instruct
#   bash struct_memory_r1/scripts/nebius_train.sh --stage 2
#   bash struct_memory_r1/scripts/nebius_train.sh --stage all --batch 8
# =============================================================================

set -euo pipefail

# Load .env if present
if [ -f "$(dirname "${BASH_SOURCE[0]}")/.env" ]; then
    set -a; source "$(dirname "${BASH_SOURCE[0]}")/.env"; set +a
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STAGE="${STAGE:-all}"              # 1 | 2 | 3 | all
NUM_GPUS="${NUM_GPUS:-1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_PROMPT_LENGTH_STAGE3="${MAX_PROMPT_LENGTH_STAGE3:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-4}"
LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-50}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
LOG_DIR="${REPO_DIR}/logs"
CHECKPOINT_DIR="${REPO_DIR}/verl_checkpoints"
DATA_DIR="${REPO_DIR}/data/struct_memory_r1"

ANSWER_AGENT_NAME="struct-r1-answer-agent-grpo"
RETRIEVE_AGENT_NAME="struct-r1-retrieve-agent-grpo"
MANAGER_NAME="struct-r1-memory-manager-grpo"

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)  STAGE="$2";           shift 2 ;;
        --gpus)   NUM_GPUS="$2";        shift 2 ;;
        --model)  BASE_MODEL="$2";      shift 2 ;;
        --steps)  TOTAL_STEPS="$2";     shift 2 ;;
        --batch)  TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/struct_r1_train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

log_section "StructMemoryR1 Training Pipeline"
log "Repo:        $REPO_DIR"
log "Stage:       $STAGE"
log "Base model:  $BASE_MODEL"
log "Num GPUs:    $NUM_GPUS"
log "Batch size:  $TRAIN_BATCH_SIZE"
log "Total steps: $TOTAL_STEPS"
log "Master log:  $MAIN_LOG"

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
log_section "Python Environment"
if conda env list 2>/dev/null | grep -q "searchr1"; then
    PYTHON="conda run --no-capture-output -n searchr1 python3"
    log "Using conda environment: searchr1"
else
    PYTHON="python3"
    log "Conda env 'searchr1' not found — using system python3"
fi

# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------
log_section "GPU Configuration"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    log "[WARN] nvidia-smi not found — assuming GPU 0"
    CUDA_VISIBLE_DEVICES="0"
fi
export CUDA_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export ATTN_BACKEND=eager
export VERL_LOG_DIR="${LOG_DIR}"

log "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# Helper: resolve model tag for experiment names
# ---------------------------------------------------------------------------
MODEL_TAG=$(echo "$BASE_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

# ---------------------------------------------------------------------------
# Build training data
# ---------------------------------------------------------------------------
build_data() {
    log_section "Building Training Data"
    if [ -f "${DATA_DIR}/answer_agent/train.parquet" ] && \
       [ -f "${DATA_DIR}/memory_manager/train.parquet" ]; then
        log "Training data already exists — skipping build."
        return 0
    fi
    log "Downloading LoCoMo and building parquet files..."
    $PYTHON -m struct_memory_r1.data.build_training_data \
        --output_dir "${DATA_DIR}" \
        2>&1 | tee "${LOG_DIR}/build_data.log"
    log "Data build complete."
}

# ---------------------------------------------------------------------------
# Memory Retrieval Server
# ---------------------------------------------------------------------------
start_memory_server() {
    log_section "Starting Memory Retrieval Server"
    $PYTHON -m struct_memory_r1.retrieval.memory_server --port 8000 &
    MEMORY_SERVER_PID=$!
    log "Memory server PID: $MEMORY_SERVER_PID"

    for i in $(seq 1 20); do
        if curl -sf http://127.0.0.1:8000/status >/dev/null 2>&1; then
            log "Memory server is ready."
            return 0
        fi
        sleep 2
    done
    log "[ERROR] Memory server did not start in time."
    exit 1
}

cleanup() {
    if [ -n "${MEMORY_SERVER_PID:-}" ]; then
        log "Stopping memory server (PID $MEMORY_SERVER_PID)..."
        kill "$MEMORY_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Common GRPO args (shared across all 3 stages)
# ---------------------------------------------------------------------------
run_grpo() {
    local DATA_TRAIN="$1"
    local DATA_VAL="$2"
    local EXPERIMENT_NAME="$3"
    local MAX_PROMPT="$4"
    local MAX_RESPONSE="$5"
    local EXTRA="$6"

    local STAGE_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"
    local CHECKPOINT_SUBDIR="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"

    log "Experiment:   $EXPERIMENT_NAME"
    log "Stage log:    $STAGE_LOG"
    log "Checkpoints:  $CHECKPOINT_SUBDIR"

    $PYTHON -m verl.trainer.main_ppo \
        data.train_files="$DATA_TRAIN" \
        data.val_files="$DATA_VAL" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.val_batch_size="${VAL_BATCH_SIZE}" \
        data.max_prompt_length="${MAX_PROMPT}" \
        data.max_response_length="${MAX_RESPONSE}" \
        data.max_start_length=512 \
        data.max_obs_length=256 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path="${BASE_MODEL}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
        actor_rollout_ref.actor.optim.warmup_style=cosine \
        actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH}" \
        actor_rollout_ref.actor.fsdp_config.param_offload=false \
        actor_rollout_ref.actor.fsdp_config.grad_offload=false \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
        actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH}" \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.kl_loss_coef=0.05 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.01 \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=4 \
        actor_rollout_ref.rollout.temperature=1.2 \
        actor_rollout_ref.actor.state_masking=false \
        trainer.logger=['console','jsonl'] \
        +trainer.val_only=false \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node="${NUM_GPUS}" \
        trainer.nnodes=1 \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.project_name=StructMemoryR1 \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.total_epochs=15 \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.default_local_dir="${CHECKPOINT_SUBDIR}" \
        do_search=false \
        max_turns=1 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        $EXTRA \
        2>&1 | tee "${STAGE_LOG}"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        log "[ERROR] Training failed (exit $EXIT_CODE). Check: $STAGE_LOG"
        exit $EXIT_CODE
    fi
    log "Training complete. Checkpoint: ${CHECKPOINT_SUBDIR}"
}

# ---------------------------------------------------------------------------
# Stage 1: Answer Agent
# ---------------------------------------------------------------------------
train_answer_agent() {
    log_section "Stage 1: Training Answer Agent"
    local EXP="${ANSWER_AGENT_NAME}-${MODEL_TAG}"
    run_grpo \
        "${DATA_DIR}/answer_agent/train.parquet" \
        "${DATA_DIR}/answer_agent/test.parquet" \
        "$EXP" \
        "${MAX_PROMPT_LENGTH}" \
        "256" \
        ""
    echo "${CHECKPOINT_DIR}/${EXP}" > "${LOG_DIR}/stage1_checkpoint.txt"
}

# ---------------------------------------------------------------------------
# Stage 2: Retrieve Agent
# ---------------------------------------------------------------------------
train_retrieve_agent() {
    local FROZEN_AA="${1:-}"
    log_section "Stage 2: Training Retrieve Agent"
    log "Frozen Answer Agent: ${FROZEN_AA:-none}"

    local EXP="${RETRIEVE_AGENT_NAME}-${MODEL_TAG}"
    local EXTRA=""
    if [ -n "$FROZEN_AA" ]; then
        EXTRA="+frozen_answer_agent_path=${FROZEN_AA}"
    fi
    run_grpo \
        "${DATA_DIR}/retrieve_agent/train.parquet" \
        "${DATA_DIR}/retrieve_agent/test.parquet" \
        "$EXP" \
        "${MAX_PROMPT_LENGTH}" \
        "512" \
        "$EXTRA"
    echo "${CHECKPOINT_DIR}/${EXP}" > "${LOG_DIR}/stage2_checkpoint.txt"
}

# ---------------------------------------------------------------------------
# Stage 3: Memory Manager
# ---------------------------------------------------------------------------
train_memory_manager() {
    local FROZEN_AA="${1:-}"
    local FROZEN_RET="${2:-}"
    log_section "Stage 3: Training Memory Manager"
    log "Frozen Answer Agent:   ${FROZEN_AA:-none}"
    log "Frozen Retrieve Agent: ${FROZEN_RET:-none}"

    local EXP="${MANAGER_NAME}-${MODEL_TAG}"
    local EXTRA=""
    if [ -n "$FROZEN_AA" ]; then
        EXTRA="+frozen_answer_agent_path=${FROZEN_AA}"
    fi
    if [ -n "$FROZEN_RET" ]; then
        EXTRA="$EXTRA +frozen_retriever_path=${FROZEN_RET}"
    fi
    run_grpo \
        "${DATA_DIR}/memory_manager/train.parquet" \
        "${DATA_DIR}/memory_manager/test.parquet" \
        "$EXP" \
        "${MAX_PROMPT_LENGTH_STAGE3}" \
        "512" \
        "$EXTRA"
    echo "${CHECKPOINT_DIR}/${EXP}" > "${LOG_DIR}/stage3_checkpoint.txt"
}

# ---------------------------------------------------------------------------
# Checkpoint resolution helpers
# ---------------------------------------------------------------------------
resolve_stage1_ckpt() {
    if [ -n "${FROZEN_ANSWER_AGENT:-}" ]; then
        echo "$FROZEN_ANSWER_AGENT"
    elif [ -f "${LOG_DIR}/stage1_checkpoint.txt" ]; then
        cat "${LOG_DIR}/stage1_checkpoint.txt"
    elif [ -d "${CHECKPOINT_DIR}/${ANSWER_AGENT_NAME}-${MODEL_TAG}" ]; then
        echo "${CHECKPOINT_DIR}/${ANSWER_AGENT_NAME}-${MODEL_TAG}"
    else
        echo ""
    fi
}

resolve_stage2_ckpt() {
    if [ -n "${FROZEN_RETRIEVER:-}" ]; then
        echo "$FROZEN_RETRIEVER"
    elif [ -f "${LOG_DIR}/stage2_checkpoint.txt" ]; then
        cat "${LOG_DIR}/stage2_checkpoint.txt"
    elif [ -d "${CHECKPOINT_DIR}/${RETRIEVE_AGENT_NAME}-${MODEL_TAG}" ]; then
        echo "${CHECKPOINT_DIR}/${RETRIEVE_AGENT_NAME}-${MODEL_TAG}"
    else
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# Setup + main
# ---------------------------------------------------------------------------
setup_check() {
    log_section "Environment Setup Check"
    $PYTHON -c "import verl; import pandas; import pyarrow; import sklearn; print('OK')" || {
        log "Installing dependencies..."
        pip install -r "${REPO_DIR}/requirements.txt"
        pip install -e "${REPO_DIR}"
        pip install scikit-learn
    }
    log "Setup check passed."
}

cd "$REPO_DIR"
setup_check
build_data
start_memory_server

case "$STAGE" in
    1)
        train_answer_agent
        ;;
    2)
        CKPT1=$(resolve_stage1_ckpt)
        [ -z "$CKPT1" ] && log "[WARN] No Stage 1 checkpoint found."
        train_retrieve_agent "$CKPT1"
        ;;
    3)
        CKPT1=$(resolve_stage1_ckpt)
        CKPT2=$(resolve_stage2_ckpt)
        [ -z "$CKPT1" ] && log "[WARN] No Stage 1 checkpoint found."
        [ -z "$CKPT2" ] && log "[WARN] No Stage 2 checkpoint found."
        train_memory_manager "$CKPT1" "$CKPT2"
        ;;
    all)
        train_answer_agent
        CKPT1=$(resolve_stage1_ckpt)
        train_retrieve_agent "$CKPT1"
        CKPT2=$(resolve_stage2_ckpt)
        train_memory_manager "$CKPT1" "$CKPT2"
        ;;
    *)
        echo "[ERROR] Unknown stage: $STAGE. Use 1, 2, 3, or all."
        exit 1
        ;;
esac

log_section "All Done"
log "Logs:        $LOG_DIR"
log "Checkpoints: $CHECKPOINT_DIR"
log "Master log:  $MAIN_LOG"
