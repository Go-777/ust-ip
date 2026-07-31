#!/bin/bash
# GRPO pipeline continuation — Round 3: further convergence observation
# Round 3: iter 7-9 (resumes from checkpoint at iter 6)
# Models: qwen3.7-max-2026-06-08(designer) + qwen-plus-2025-07-28(executor/judge) + qwen-flash(selector/qa)
# Note: judge model change triggers automatic baseline reset in code
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen-flash \
    --selector-model qwen-flash \
    --executor-model qwen-plus-2025-07-28 \
    --qa-model qwen-flash \
    --designer-model qwen3.7-max-2026-06-08 \
    --judge-model qwen-plus-2025-07-28 \
    --grpo-group-size 4 \
    --grpo-max-iterations 50 \
    --grpo-num-bad-cases 20 \
    --grpo-case-chunk-size 5 \
    --grpo-early-stop-patience 50 \
    --grpo-early-stop-warmup 50 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge