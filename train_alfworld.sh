#!/bin/bash
# Small-batch GRPO test run — validate pipeline correctness
# Models: qwen3.7-max(designer) + qwen-plus-latest(executor/judge) + qwen-flash(selector/qa)
# Config: 5 cases, chunk=5, G=4, 2 iterations (~免费额度)
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen-flash \
    --selector-model qwen-flash \
    --executor-model qwen-plus-latest \
    --qa-model qwen-flash \
    --designer-model qwen3.7-max \
    --judge-model qwen-plus-latest \
    --grpo-group-size 2 \
    --grpo-max-iterations 1 \
    --grpo-num-bad-cases 3 \
    --grpo-case-chunk-size 3 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge