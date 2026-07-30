#!/bin/bash
# MemSkill GRPO 实验 — 小批量验证 (免费额度模型)
# 模型分配策略: 用额度充足的模型，避免403/503
# Designer=qwen3.7-max-preview (1M, 100%), Judge=qwen3.7-max-2026-05-17 (954K)
# Executor=qwen3.6-plus (1M, 100%), Selector=qwen3.6-flash (1M, 100%)
# 验证目标: 2-3 iterations确认代码流程正确、reward有产出
# 验证通过后切换到大规模配置

export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen3.6-flash \
    --selector-model qwen3.6-flash \
    --executor-model qwen3.6-plus \
    --designer-model qwen3.7-max-preview \
    --judge-model qwen3.7-max-2026-05-17 \
    --grpo-group-size 4 \
    --grpo-max-iterations 3 \
    --grpo-num-bad-cases 10 \
    --grpo-case-chunk-size 5 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge