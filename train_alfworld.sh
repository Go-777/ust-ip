#!/bin/bash
# GRPO pipeline continuation — observe convergence with fresh model quota
# Round 2: iter 4-6 (resumes from checkpoint at iter 3)
# Models: qwen3.6-max-preview(designer) +qwen-plus-2025-09-11(executor/judge) + qwen-flash-2025-07-28(selector/qa)
# Note: judge model change triggers automatic baseline reset in code
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen-flash-2025-07-28 \
    --selector-model qwen-flash-2025-07-28 \
    --executor-model qwen-plus-2025-09-11 \
    --qa-model qwen-flash-2025-07-28 \
    --designer-model qwen3.6-max-preview \
    --judge-model qwen-plus-2025-09-11 \
    --grpo-group-size 4 \
    --grpo-max-iterations 6 \
    --grpo-num-bad-cases 20 \
    --grpo-case-chunk-size 5 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge