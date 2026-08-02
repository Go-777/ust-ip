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
# Model allocation (updated — use highest-quota models):
#   designer  → qwen3.7-max-2026-05-17     (85.86% quota, high capability for design)
#   selector  → qwen3.7-plus-2026-05-26    (~65% quota)
#   executor  → qwen3.6-max-preview         (~30% quota, good instruction following)
#   qa        → qwen3.7-max-preview         (~45% quota)
#   judge     → qwen3.6-plus-2026-04-02    (100% quota! simple scoring task, plus is sufficient)
#
# EXHAUSTED (do NOT use): qwen3.7-max-2026-06-08, qwen3-max, qwen-plus-2025-12-01
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
    --model qwen3.7-plus-2026-05-26 \
    --selector-model qwen3.7-plus-2026-05-26 \
    --executor-model qwen3.6-max-preview \
    --qa-model qwen3.7-max-preview \
    --designer-model qwen3.7-max-2026-05-17 \
    --judge-model qwen3.6-plus-2026-04-02 \
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