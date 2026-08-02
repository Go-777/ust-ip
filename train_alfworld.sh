#!/bin/bash
# GRPO pipeline — production run (scale up from verification)
#
# Previous verification run (3 iters) result:
#   - Iter 1: avg reward 0.8646 (best), Iter 2: 0.8333 (partial), Iter 3: 0.0 (designer exhausted)
#   - Pipeline validated: 3 samples exported, logic correct
#   - Issue: designer model qwen3.7-max-2026-06-08 quota exhausted
#
# This run: swap designer to qwen3.7-max-2026-05-17 (85.86%), judge to qwen3.6-plus-2026-04-02 (100%), scale to 30 iters
# Target: produce 50+ GRPO training samples
#
# Model allocation (maximized — all high-quota models):
#   designer  → qwen3-max-preview           (100% quota, 1M tokens, high capability)
#   selector  → qwen3.5-flash              (81.45% quota, fast and cheap for selection)
#   executor  → qwen3-max-2026-01-23       (100% quota, 1M tokens, good instruction following)
#   qa        → qwen3-max                  (80.68% quota, reading comprehension)
#   judge     → qwen3.6-flash-2026-04-16   (100% quota, simple scoring, flash is sufficient)
#
# RESERVED (not used this run): qwen3.7-max-2026-05-17(85.86%), qwen3.7-max-2026-05-20(65.8%)
# EXHAUSTED (do NOT use): qwen3.7-max-2026-06-08, qwen-plus-2025-12-01
#
# Estimated budget (30 iters, 5 cases, chunk=3, group=4):
#   Designer: ~10 calls/iter × 30 = ~300 calls (~600K tokens)
#   Others: ~80 calls/iter × 30 = ~2400 calls (~2.4M tokens)
#   Estimated time: ~3-4 hours (API latency dominated)
#
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"
export PYTHONUNBUFFERED=1

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --save-dir "./checkpoints_grpo" \
    --model qwen3-max-preview \
    --selector-model qwen3.5-flash \
    --executor-model qwen3-max-2026-01-23 \
    --qa-model qwen3-max \
    --designer-model qwen3-max-preview \
    --judge-model qwen3.6-flash-2026-04-16 \
    --grpo-group-size 4 \
    --grpo-max-iterations 30 \
    --grpo-num-bad-cases 5 \
    --grpo-case-chunk-size 3 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge