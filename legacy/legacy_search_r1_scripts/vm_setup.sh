#!/bin/bash
# =============================================================================
# Memory-R1 VM Setup Script — Run this ONCE after SSH-ing into the Nebius VM
#
# Usage:
#   bash vm_setup.sh
#
# What it does:
#   1. Installs system packages (CUDA tools, git, tmux, htop)
#   2. Installs Miniconda and creates the 'searchr1' environment
#   3. Clones the Search-R1 repo
#   4. Installs all Python dependencies (flash-attn, vllm, verl, etc.)
#   5. Verifies the GPU and imports
# =============================================================================

set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
CONDA_ENV="searchr1"
REPO_DIR="$HOME/Search-R1"
REPO_URL="https://github.com/david-courtis/Search-R1.git"

LOG_FILE="$HOME/vm_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log()     { echo "[$(date '+%H:%M:%S')] $*"; }
section() { echo ""; echo "==== $* ===="; }

section "Nebius VM Setup for Memory-R1"
log "Log: $LOG_FILE"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
section "System Packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    git curl wget tmux htop nvtop \
    build-essential libssl-dev \
    python3-dev
log "System packages installed."

# ---------------------------------------------------------------------------
# 2. Miniconda
# ---------------------------------------------------------------------------
section "Miniconda"
if [ ! -d "$CONDA_DIR" ]; then
    log "Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    log "Miniconda installed to $CONDA_DIR"
else
    log "Miniconda already installed."
fi

# Initialise conda for this script's shell session
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# Add to .bashrc for future logins
if ! grep -q "conda shell.bash hook" ~/.bashrc; then
    echo 'eval "$('"$CONDA_DIR"'/bin/conda shell.bash hook)"' >> ~/.bashrc
fi

# ---------------------------------------------------------------------------
# 3. Clone repo
# ---------------------------------------------------------------------------
section "Clone Search-R1 Repo"
if [ ! -d "$REPO_DIR/.git" ]; then
    log "Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "Repo already cloned. Pulling latest..."
    git -C "$REPO_DIR" pull
fi

# ---------------------------------------------------------------------------
# 4. Conda environment
# ---------------------------------------------------------------------------
section "Conda Environment: $CONDA_ENV"
if conda env list | grep -q "^${CONDA_ENV}"; then
    log "Environment '$CONDA_ENV' already exists."
else
    log "Creating conda env '$CONDA_ENV' with Python 3.10..."
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

conda activate "$CONDA_ENV"
log "Active env: $(which python)"

# ---------------------------------------------------------------------------
# 5. Python dependencies
# ---------------------------------------------------------------------------
section "Python Dependencies"
cd "$REPO_DIR"

log "Installing core requirements..."
pip install --quiet --no-cache-dir \
    accelerate codetiming datasets dill hydra-core \
    numpy pandas pybind11 ray \
    "tensordict<0.6" \
    "transformers<4.48" \
    "vllm<=0.6.3" \
    wandb scikit-learn pyarrow IPython matplotlib

log "Installing flash-attn (compiles from source — takes ~5 min)..."
pip install --quiet --no-cache-dir flash-attn --no-build-isolation

log "Installing Search-R1 in editable mode..."
pip install --quiet -e .

log "Python dependencies installed."

# ---------------------------------------------------------------------------
# 6. Verify GPU
# ---------------------------------------------------------------------------
section "GPU Verification"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
conda run -n "$CONDA_ENV" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} — {props.total_memory // 1024**3} GB')
"

# ---------------------------------------------------------------------------
# 7. Quick import test
# ---------------------------------------------------------------------------
section "Import Tests"
conda run -n "$CONDA_ENV" python3 -c "
import verl, pandas, pyarrow, sklearn, vllm
print('All imports OK: verl, pandas, pyarrow, sklearn, vllm')
"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
section "Setup Complete"
log "Repo:    $REPO_DIR"
log "Env:     conda activate $CONDA_ENV"
log "Log:     $LOG_FILE"
log ""
log "Next step: copy nebius_train_memory_r1.sh to $REPO_DIR/ then:"
log "  cd $REPO_DIR"
log "  conda activate $CONDA_ENV"
log "  bash nebius_train_memory_r1.sh"
