#!/bin/bash
# MemSkill GRPO 小批量完整探索实验 (v4 - evolution management)
# 修复: 
#   v3: record_usage -> update_stats
#   v4: + threshold-based apply + parse retry + evolution_history + skill_usage_stats
# 参数: 5轮, group_size=4, 6 bad_cases, chunk_size=3, temp=0.9
# 模型: qwen3.7-flash(主) + 各角色独立模型(分散额度)

export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"

python train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen3.7-flash \
    --selector-model qwen3.7-flash-2026-07-15 \
    --executor-model qwen3.5-plus-2026-04-20 \
    --designer-local-model qwen3.7-plus-2026-05-26 \
    --judge-model qwen3.7-max-2026-06-08 \
    --grpo-group-size 4 \
    --grpo-max-iterations 5 \
    --grpo-num-bad-cases 6 \
    --grpo-case-chunk-size 3 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge