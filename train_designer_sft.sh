#!/bin/bash
# ============================================================
# MemSkill Designer SFT Fine-tuning via Reject Sampling
# Model: Qwen2.5-7B-Instruct + QLoRA
# Data: 25 samples (best response per group, reward >= 0.3)
# ============================================================
set -e

echo "=== MemSkill Designer SFT Training ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "======================================="

# Set project directory
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# Create output dirs
mkdir -p checkpoints/designer_sft logs

# Step 1: Prepare SFT data (reject sampling)
echo ""
echo "[Step 1/2] Preparing SFT training data..."
python scripts/prepare_sft_data.py \
    --input grpo_data/grpo_training_data_filtered.jsonl \
    --output grpo_data/sft_train.jsonl \
    --top-k 1 \
    --min-reward 0.3

# Step 2: Run SFT training
echo ""
echo "[Step 2/2] Running SFT training..."
python train_sft_designer.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --data_path grpo_data/sft_train.jsonl \
    --output_dir checkpoints/designer_sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_seq_length 8192 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_steps 10 \
    2>&1 | tee logs/designer_sft_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Output: checkpoints/designer_sft/"