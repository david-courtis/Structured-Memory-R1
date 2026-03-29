# Model Checkpoints

This directory contains the trained model checkpoints for StructMemoryR1 and the Memory-R1 baseline. Due to file size constraints, checkpoint weights are not included in this repository and will be released via Google Drive in the final submission.

## Directory Structure

```
checkpoints/
├── struct_memory_r1/
│   ├── answer_agent/          # Stage 1: Answer Agent (Qwen2.5-3B-Instruct, 500 steps)
│   ├── retrieve_agent/        # Stage 2: Retrieve Agent (Qwen2.5-3B-Instruct, 500 steps)
│   └── memory_manager/        # Stage 3: Memory Manager (Qwen2.5-3B-Instruct, 500 steps)
└── memory_r1/
    ├── answer_agent/          # Stage 1: Answer Agent (Qwen2.5-3B-Instruct, 500 steps)
    └── memory_manager/        # Stage 2: Memory Manager (Qwen2.5-3B-Instruct, 500 steps)
```

## StructMemoryR1 Checkpoints

Three-stage GRPO-trained agents on LoCoMo with Qwen2.5-3B-Instruct as the base model:

| Agent | Stage | Training Steps | Description |
|-------|-------|---------------|-------------|
| Answer Agent | 1 | 500 | Trained with token F1 reward; frozen in Stages 2-3 |
| Retrieve Agent | 2 | 500 | Trained with F1 + evidence hit reward; frozen Answer Agent as evaluator |
| Memory Manager | 3 | 500 | Trained with F1 + JSON validity reward; frozen Answer Agent + Retrieve Agent |

## Memory-R1 Checkpoints (Baseline)

Two-stage GRPO-trained agents (our re-implementation of Memory-R1):

| Agent | Stage | Training Steps | Description |
|-------|-------|---------------|-------------|
| Answer Agent | 1 | 500 | Trained with token F1 reward; frozen in Stage 2 |
| Memory Manager | 2 | 500 | Trained with F1 + JSON validity reward; frozen Answer Agent as evaluator |

## Hardware

All checkpoints were trained on a single NVIDIA H200 GPU (141 GB HBM3e). Total training time across all checkpoints is approximately 150 GPU-hours.

## Usage

To load a checkpoint for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/struct_memory_r1/answer_agent")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/struct_memory_r1/answer_agent")
```

To use a checkpoint as a frozen agent during training:

```bash
FROZEN_ANSWER_AGENT=checkpoints/struct_memory_r1/answer_agent \
bash struct_memory_r1/scripts/train_retrieve_agent.sh
```

## Download

**Checkpoints will be released as a Google Drive link in the final submission due to repository size constraints.**
