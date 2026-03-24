#!/bin/bash
# =============================================================================
# Memory-R1: Full Training Pipeline for Nebius Cloud (GPU Cluster)
#
# Usage:
#   bash nebius_train_memory_r1.sh [--stage <1|2|both>] [--gpus <N>] [--model <hf_model>]
#
# Stages:
#   1    - Train Answer Agent only (Stage 1)
#   2    - Train Memory Manager only (Stage 2, requires Stage 1 checkpoint)
#   both - Run Stage 1 then Stage 2 sequentially (default)
#
# Examples:
#   bash nebius_train_memory_r1.sh
#   bash nebius_train_memory_r1.sh --stage 1 --gpus 4
#   bash nebius_train_memory_r1.sh --stage 2 --gpus 4
#   bash nebius_train_memory_r1.sh --stage both --model Qwen/Qwen2.5-1.5B
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Default configuration — override via CLI flags or environment variables
# ---------------------------------------------------------------------------
STAGE="${STAGE:-both}"           # 1 | 2 | both
NUM_GPUS="${NUM_GPUS:-1}"        # Number of GPUs on this node
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"       # 0.6 for 7B on H200 141GB; can raise to 0.7
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-50}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_DIR}/logs"
CHECKPOINT_DIR="${REPO_DIR}/verl_checkpoints"
DATA_DIR="${REPO_DIR}/data/memory_r1"

ANSWER_AGENT_NAME="memory-r1-answer-agent-grpo"
MANAGER_NAME="memory-r1-manager-grpo"

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)  STAGE="$2";     shift 2 ;;
        --gpus)   NUM_GPUS="$2";  shift 2 ;;
        --model)  BASE_MODEL="$2"; shift 2 ;;
        --steps)  TOTAL_STEPS="$2"; shift 2 ;;
        --batch)  TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

MAIN_LOG="${LOG_DIR}/nebius_train_$(date +%Y%m%d_%H%M%S).log"

# Tee everything to both stdout and the master log
exec > >(tee -a "$MAIN_LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_section() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

log_section "Memory-R1 Nebius Training Job Started"
log "Repo:        $REPO_DIR"
log "Stage:       $STAGE"
log "Base model:  $BASE_MODEL"
log "Num GPUs:    $NUM_GPUS"
log "Batch size:  $TRAIN_BATCH_SIZE"
log "Total steps: $TOTAL_STEPS"
log "Master log:  $MAIN_LOG"

# ---------------------------------------------------------------------------
# Detect Python: prefer conda env 'searchr1', fall back to system python3
# ---------------------------------------------------------------------------
log_section "Python Environment"
if conda env list 2>/dev/null | grep -q "searchr1"; then
    PYTHON="conda run --no-capture-output -n searchr1 python3"
    log "Using conda environment: searchr1"
else
    PYTHON="python3"
    log "Conda env 'searchr1' not found — using system python3"
    log "[INFO] Install dependencies with: pip install -r requirements.txt && pip install -e ."
fi

# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------
log_section "GPU Configuration"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    # Build CUDA_VISIBLE_DEVICES string: 0,1,2,...,N-1
    CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    log "[WARN] nvidia-smi not found — assuming GPU 0"
    CUDA_VISIBLE_DEVICES="0"
fi
export CUDA_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # Use FlashAttention on A100/H100
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export ATTN_BACKEND=eager                  # HF model init on CPU before moving to GPU

log "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
log "VLLM_ATTENTION_BACKEND = $VLLM_ATTENTION_BACKEND"

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
    $PYTHON -m memory_r1.data.build_training_data \
        --output_dir "${DATA_DIR}" \
        2>&1 | tee "${LOG_DIR}/build_data.log"
    log "Data build complete. Log: ${LOG_DIR}/build_data.log"
}

# ---------------------------------------------------------------------------
# Stage 1: Train Answer Agent
# ---------------------------------------------------------------------------
train_answer_agent() {
    log_section "Stage 1: Training Answer Agent"

    local EXPERIMENT_NAME="${ANSWER_AGENT_NAME}-$(echo "$BASE_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
    local STAGE_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"
    local CHECKPOINT_SUBDIR="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"

    log "Experiment:   $EXPERIMENT_NAME"
    log "Stage log:    $STAGE_LOG"
    log "Checkpoints:  $CHECKPOINT_SUBDIR"

    # Progress tracking file — updated by the tee'd log
    local PROGRESS_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_progress.txt"
    echo "stage=1 experiment=${EXPERIMENT_NAME} started=$(date '+%Y-%m-%d %H:%M:%S')" > "$PROGRESS_FILE"

    # Launch training; pipe through awk to emit progress to progress file
    $PYTHON -m verl.trainer.main_ppo \
        data.train_files="${DATA_DIR}/answer_agent/train.parquet" \
        data.val_files="${DATA_DIR}/answer_agent/test.parquet" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.val_batch_size="${VAL_BATCH_SIZE}" \
        data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
        data.max_response_length=256 \
        data.max_start_length=512 \
        data.max_obs_length=256 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path="${BASE_MODEL}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH}" \
        actor_rollout_ref.actor.fsdp_config.param_offload=false \
        actor_rollout_ref.actor.fsdp_config.grad_offload=false \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
        actor_rollout_ref.rollout.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=2 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.actor.state_masking=false \
        trainer.logger=['console'] \
        +trainer.val_only=false \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node="${NUM_GPUS}" \
        trainer.nnodes=1 \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.project_name=Memory-R1 \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.total_epochs=15 \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.default_local_dir="${CHECKPOINT_SUBDIR}" \
        do_search=false \
        max_turns=1 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        2>&1 | tee "${STAGE_LOG}" | awk '
            /step.*reward|Step.*Reward|global_step/ {
                print strftime("[%Y-%m-%d %H:%M:%S]"), "PROGRESS:", $0
            }
            { print }
        '

    local EXIT_CODE=${PIPESTATUS[0]}
    echo "stage=1 finished=$(date '+%Y-%m-%d %H:%M:%S') exit_code=${EXIT_CODE}" >> "$PROGRESS_FILE"

    if [ $EXIT_CODE -ne 0 ]; then
        log "[ERROR] Stage 1 training failed (exit $EXIT_CODE). Check: $STAGE_LOG"
        exit $EXIT_CODE
    fi
    log "Stage 1 complete. Checkpoint: ${CHECKPOINT_SUBDIR}"
    # Write checkpoint path to file so the caller can read it without capturing stdout
    echo "$CHECKPOINT_SUBDIR" > "${LOG_DIR}/stage1_checkpoint.txt"
}

# ---------------------------------------------------------------------------
# Stage 2: Train Memory Manager
# ---------------------------------------------------------------------------
train_memory_manager() {
    local FROZEN_AGENT="${1:-}"
    log_section "Stage 2: Training Memory Manager"

    local EXPERIMENT_NAME="${MANAGER_NAME}-$(echo "$BASE_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
    local STAGE_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"
    local CHECKPOINT_SUBDIR="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"

    log "Experiment:       $EXPERIMENT_NAME"
    log "Stage log:        $STAGE_LOG"
    log "Checkpoints:      $CHECKPOINT_SUBDIR"
    log "Frozen Answer Ag: ${FROZEN_AGENT:-none (format-only reward)}"

    local PROGRESS_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_progress.txt"
    echo "stage=2 experiment=${EXPERIMENT_NAME} started=$(date '+%Y-%m-%d %H:%M:%S')" > "$PROGRESS_FILE"

    EXTRA_ARGS=""
    if [ -n "$FROZEN_AGENT" ]; then
        EXTRA_ARGS="+frozen_answer_agent_path=${FROZEN_AGENT}"
    fi

    $PYTHON -m verl.trainer.main_ppo \
        data.train_files="${DATA_DIR}/memory_manager/train.parquet" \
        data.val_files="${DATA_DIR}/memory_manager/test.parquet" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.val_batch_size="${VAL_BATCH_SIZE}" \
        data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
        data.max_response_length=512 \
        data.max_start_length=512 \
        data.max_obs_length=256 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path="${BASE_MODEL}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH}" \
        actor_rollout_ref.actor.fsdp_config.param_offload=false \
        actor_rollout_ref.actor.fsdp_config.grad_offload=false \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
        actor_rollout_ref.rollout.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=2 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.actor.state_masking=false \
        trainer.logger=['console'] \
        +trainer.val_only=false \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node="${NUM_GPUS}" \
        trainer.nnodes=1 \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.project_name=Memory-R1 \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.total_epochs=15 \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.default_local_dir="${CHECKPOINT_SUBDIR}" \
        do_search=false \
        max_turns=1 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        $EXTRA_ARGS \
        2>&1 | tee "${STAGE_LOG}" | awk '
            /step.*reward|Step.*Reward|global_step/ {
                print strftime("[%Y-%m-%d %H:%M:%S]"), "PROGRESS:", $0
            }
            { print }
        '

    local EXIT_CODE=${PIPESTATUS[0]}
    echo "stage=2 finished=$(date '+%Y-%m-%d %H:%M:%S') exit_code=${EXIT_CODE}" >> "$PROGRESS_FILE"

    if [ $EXIT_CODE -ne 0 ]; then
        log "[ERROR] Stage 2 training failed (exit $EXIT_CODE). Check: $STAGE_LOG"
        exit $EXIT_CODE
    fi
    log "Stage 2 complete. Checkpoint: ${CHECKPOINT_SUBDIR}"
}

# ---------------------------------------------------------------------------
# Environment setup check
# ---------------------------------------------------------------------------
setup_check() {
    log_section "Environment Setup Check"
    log "Verifying Python dependencies..."
    $PYTHON -c "import verl; import pandas; import pyarrow; import sklearn; print('Core imports OK')" || {
        log "[INFO] Installing dependencies..."
        pip install -r "${REPO_DIR}/requirements.txt"
        pip install -e "${REPO_DIR}"
        pip install scikit-learn
    }
    log "Setup check passed."
}

# ---------------------------------------------------------------------------
# Memory Retrieval Server (required during training rollouts)
# ---------------------------------------------------------------------------
start_memory_server() {
    log_section "Starting Memory Retrieval Server"
    $PYTHON -m memory_r1.retrieval.memory_server --port 8000 &
    MEMORY_SERVER_PID=$!
    log "Memory server PID: $MEMORY_SERVER_PID"

    # Wait for it to be ready
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
# Main
# ---------------------------------------------------------------------------
cd "$REPO_DIR"

setup_check
build_data
start_memory_server

ANSWER_AGENT_CKPT=""

case "$STAGE" in
    1)
        train_answer_agent
        ;;
    2)
        # Look for existing best checkpoint from Stage 1
        MODEL_TAG=$(echo "$BASE_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
        CANDIDATE="${CHECKPOINT_DIR}/${ANSWER_AGENT_NAME}-${MODEL_TAG}/best"
        if [ -n "${FROZEN_ANSWER_AGENT:-}" ]; then
            log "Using FROZEN_ANSWER_AGENT from env: $FROZEN_ANSWER_AGENT"
            ANSWER_AGENT_CKPT="$FROZEN_ANSWER_AGENT"
        elif [ -d "$CANDIDATE" ]; then
            log "Found Stage 1 checkpoint: $CANDIDATE"
            ANSWER_AGENT_CKPT="$CANDIDATE"
        elif [ -f "${LOG_DIR}/stage1_checkpoint.txt" ]; then
            ANSWER_AGENT_CKPT=$(cat "${LOG_DIR}/stage1_checkpoint.txt")
            log "Found Stage 1 checkpoint from log: $ANSWER_AGENT_CKPT"
        else
            log "[WARN] No Stage 1 checkpoint found. Running Stage 2 with format-only reward."
            ANSWER_AGENT_CKPT=""
        fi
        train_memory_manager "$ANSWER_AGENT_CKPT"
        ;;
    both)
        train_answer_agent
        # Find best checkpoint from Stage 1
        MODEL_TAG=$(echo "$BASE_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
        BEST_CKPT="${CHECKPOINT_DIR}/${ANSWER_AGENT_NAME}-${MODEL_TAG}/best"
        if [ -d "$BEST_CKPT" ]; then
            ANSWER_AGENT_CKPT="$BEST_CKPT"
        elif [ -f "${LOG_DIR}/stage1_checkpoint.txt" ]; then
            ANSWER_AGENT_CKPT=$(cat "${LOG_DIR}/stage1_checkpoint.txt")
        else
            ANSWER_AGENT_CKPT=""
        fi
        train_memory_manager "$ANSWER_AGENT_CKPT"
        ;;
    *)
        echo "[ERROR] Unknown stage: $STAGE. Use 1, 2, or both."
        exit 1
        ;;
esac

log_section "All Done"
log "Logs:        $LOG_DIR"
log "Checkpoints: $CHECKPOINT_DIR"
log "Master log:  $MAIN_LOG"
