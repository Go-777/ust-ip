#!/bin/bash
# 本地CPU验证: 使用百炼API (1M tokens免费额度) 跑通GRPO全链路
# 目标: 验证端到端pipeline逻辑，1次迭代，group_size=2，预计消耗 ~5K tokens
#
# 百炼API: https://dashscope.aliyuncs.com/compatible-mode/v1
# 模型: qwen3.7-plus (性价比高, 1M免费额度)

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

echo "=== GRPO Local CPU Verify: 1 iteration, group_size=2 ==="
echo "Using: Dashscope API (qwen3.7-plus, 1M free tokens)"
echo "Start time: $(date)"
echo ""

python train_grpo.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --bad-cases-file "./data/bad_cases_mini.json" \
    --model qwen3.7-plus \
    --api --api-base "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api-key "${DASHSCOPE_API_KEY}" \
    --grpo-enabled \
    --grpo-group-size 2 \
    --grpo-max-iterations 1 \
    --grpo-num-bad-cases 4 \
    --grpo-case-chunk-size 2 \
    --grpo-temperature 0.7 \
    --grpo-max-designer-tokens 2048 \
    --action-top-k 2 \
    --max-new-tokens 2048 \
    --device cpu \
    --save-dir ./checkpoints/grpo_local_verify \
    --grpo-export-dir ./grpo_data/local_verify \
    2>&1 | tee logs/grpo_local_verify_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "End time: $(date)"
echo "=== Done ==="