#!/bin/bash
# MemSkill GRPO 正式运行脚本（登录节点，tmux中执行）
# 用法: tmux new -s grpo && bash run_grpo.sh

cd ~/ust-ip

eval "$(conda shell.bash hook)"
conda activate memskill

export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
# API Key: set via environment or uncomment below
# export DASHSCOPE_API_KEY='your-key-here'
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"

mkdir -p logs checkpoints/grpo_full grpo_data/full

echo "========================================"
echo "  MemSkill GRPO Full Run"
echo "  Start: $(date)"
echo "  Reward: llm_judge | Data: bad_cases_extended"
echo "========================================"

python train_grpo.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen-plus \
    --api --api-base "${DASHSCOPE_API_BASE}" \
    --grpo-enabled \
    --grpo-group-size 8 \
    --grpo-max-iterations 20 \
    --grpo-num-bad-cases 8 \
    --grpo-case-chunk-size 4 \
    --grpo-temperature 0.7 \
    --grpo-max-designer-tokens 2048 \
    --reward-metric llm_judge \
    --action-top-k 3 \
    --max-new-tokens 2048 \
    --device cpu \
    --save-dir ./checkpoints/grpo_full \
    --grpo-export-dir ./grpo_data/full \
    2>&1 | tee logs/grpo_full_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "[Done] End: $(date)"