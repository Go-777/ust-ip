#!/bin/bash
# GRPO pipeline — verification run (small scale to validate pipeline works)
# Then scale up: --grpo-max-iterations 50 --grpo-group-size 8
# Models: qwen3-max(designer) + qwen-plus-2025-12-01(executor/judge/selector/qa)
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --save-dir "./checkpoints_verify" \
    --model qwen-plus-2025-12-01 \
    --selector-model qwen-plus-2025-12-01 \
    --executor-model qwen-plus-2025-12-01 \
    --qa-model qwen-plus-2025-12-01 \
    --designer-model qwen3-max \
    --judge-model qwen-plus-2025-12-01 \
    --grpo-group-size 4 \
    --grpo-max-iterations 3 \
    --grpo-num-bad-cases 5 \
    --grpo-case-chunk-size 3 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge