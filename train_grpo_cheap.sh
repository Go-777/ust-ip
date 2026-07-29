#!/bin/bash
# MemSkill GRPO 实验 (接近原方案效果，Judge降一级)
# 原方案: Selector/Executor=Plus, Designer=本地9B, Judge=Max
# 当前方案: Selector=Flash, Executor=Plus, Designer=Plus, Judge=Plus
# 变化: Judge Max→Plus (省最多); Selector Plus→Flash (任务简单)
# 预估成本: ¥100-150 (全量50 iterations)
# 预估时间: 4-6小时

export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"

python train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen3.5-flash \
    --selector-model qwen3.5-flash \
    --executor-model qwen3.5-plus \
    --designer-local-model qwen3.7-plus \
    --judge-model qwen3.5-plus \
    --grpo-group-size 4 \
    --grpo-max-iterations 50 \
    --grpo-num-bad-cases 20 \
    --grpo-case-chunk-size 5 \
    --grpo-early-stop-patience 15 \
    --grpo-early-stop-warmup 10 \
    --grpo-temperature 0.9 \
    --grpo-min-apply-threshold 0.3 \
    --grpo-max-parse-retries 1 \
    --reward-metric llm_judge