#!/bin/bash
# GRPO pipeline — verification run (small scale to validate pipeline works)
# Then scale up: --grpo-max-iterations 50 --grpo-group-size 8
# Models: qwen3.7-max-2026-06-08(designer) + qwen3.7-plus-2026-05-26(others)
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --save-dir "./checkpoints_verify" \
    --model qwen3.7-plus-2026-05-26 \
    --selector-model qwen3.7-plus-2026-05-26 \
    --executor-model qwen3.7-plus-2026-05-26 \
    --qa-model qwen3.7-plus-2026-05-26 \
    --designer-model qwen3.7-max-2026-06-08 \
    --judge-model qwen3.7-plus-2026-05-26 \
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