#!/bin/bash
# MemSkill GRPO 实验 — 小批量验证 (免费额度模型)
# Judge=qwen3.7-max (最强, 1M免费), Designer=qwen3.7-max (1M免费)
# Executor=qwen3.5-plus (1M免费), Selector=qwen3.5-flash (1M免费)
# 验证目标: 2-3 iterations确认代码流程正确、reward有产出
# 验证通过后切换到大规模配置

export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"

python train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen3.5-flash-2026-02-23 \
    --selector-model qwen3.5-flash-2026-02-23 \
    --executor-model qwen3.5-plus-2026-02-15 \
    --designer-model qwen3.7-max-2026-05-20 \
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