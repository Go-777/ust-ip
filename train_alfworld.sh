#!/bin/bash
# GRPO pipeline — verification run (small scale to validate pipeline works)
# Then scale up: --grpo-max-iterations 50 --grpo-group-size 8
#
# Token budget estimation (3 iterations, 5 cases, chunk=3, group=4):
#   Per iteration: ~2 designer(analysis) + ~8 designer(sampling) + 20 each for selector/executor/qa/judge
#   Total: ~270 API calls across 3 iterations
#   Designer (max model): ~10 calls/iter × ~2K tok = ~60K tokens total
#   Selector/Executor/Judge/QA (plus models): ~80 calls/iter × ~1K tok = ~240K tokens total
#   Estimated time: ~15-20 min (API latency dominated)
#
# Model allocation (spread across models to avoid exhausting any single one):
#   designer  → qwen3.7-max-2026-06-08    (highest capability, ~10 calls/iter)
#   selector  → qwen3.7-plus-2026-05-26   (20 calls/iter, simple selection)
#   executor  → qwen3.6-max-preview        (20 calls/iter, needs good instruction following)
#   qa        → qwen3.7-max-preview        (20 calls/iter, reading comprehension)
#   judge     → qwen3.7-max-2026-05-20     (20 calls/iter, scoring)
#
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --save-dir "./checkpoints_verify" \
    --model qwen3.7-plus-2026-05-26 \
    --selector-model qwen3.7-plus-2026-05-26 \
    --executor-model qwen3.6-max-preview \
    --qa-model qwen3.7-max-preview \
    --designer-model qwen3.7-max-2026-06-08 \
    --judge-model qwen3.7-max-2026-05-20 \
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