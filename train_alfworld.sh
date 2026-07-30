#!/bin/bash
# GRPO pipeline validation run — target ~800K tokens per model (80% of 1M free quota)
# Models: qwen3.7-max(designer) + qwen-plus-latest(executor/judge) + qwen-flash(selector/qa)
# Token budget estimate:
#   designer(max):  3iter × 4chunks × ~10K = ~120K tokens
#   executor+judge(plus): 3×4×20 × 3.1K = ~756K tokens
#   selector+qa(flash):   3×4×20 × 2.8K = ~672K tokens
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
    --grpo-group-size 4 \
    --grpo-max-iterations 3 \
    --grpo-num-bad-cases 20 \
    --grpo-case-chunk-size 5 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge