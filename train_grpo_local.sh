#!/bin/bash
# ============================================================
# GRPO 本地模型验证脚本 — 使用 vLLM 本地部署的 Qwen2.5-7B
#
# 前置条件: 
#   1. 已运行 scripts/deploy_vllm_local.sh 启动 vLLM
#   2. data/bad_cases_100.json 已生成
#
# 用法:
#   ssh root@nb-m7wkdajepx
#   cd /mnt/workspace/home/zhaozhichen1/MemSkill
#   bash train_grpo_local.sh
#
# 特点:
#   - 所有角色统一使用本地 Qwen2.5-7B（无需 DashScope 额度）
#   - group_size=4 减少推理量，加快验证
#   - 10次迭代验证完整流程
# ============================================================

set -e

# === 环境 ===
export PATH="/mnt/workspace/envs/vllm-py312/bin:$PATH"
export PYTHONUNBUFFERED=1

# === 本地 vLLM 配置 ===
VLLM_ENDPOINT="http://localhost:8000/v1"
VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
# vLLM 本地部署无需真实 API key，但 OpenAI client 需要非空值
export DASHSCOPE_API_KEY="local-vllm-no-key-needed"

# === 验证 vLLM 是否在运行 ===
echo "Checking vLLM server..."
if ! curl -s ${VLLM_ENDPOINT}/models > /dev/null 2>&1; then
    echo "ERROR: vLLM server not running at ${VLLM_ENDPOINT}"
    echo "Please run: bash scripts/deploy_vllm_local.sh"
    exit 1
fi
echo "  ✓ vLLM is running"

# === 开始 GRPO 训练 ===
mkdir -p checkpoints/grpo_local grpo_data/local logs

echo ""
echo "============================================"
echo "  GRPO Training with Local vLLM"
echo "  Model: ${VLLM_MODEL}"
echo "  Data: 100 bad cases"
echo "  Iterations: 10"
echo "  Start: $(date)"
echo "============================================"
echo ""

python -u train_grpo.py \
    --grpo-enabled \
    --bad-cases-file "./data/bad_cases_100.json" \
    --save-dir "./checkpoints/grpo_local" \
    --grpo-export-dir "./grpo_data/local" \
    --model "${VLLM_MODEL}" \
    --selector-model "${VLLM_MODEL}" \
    --executor-model "${VLLM_MODEL}" \
    --qa-model "${VLLM_MODEL}" \
    --designer-model "${VLLM_MODEL}" \
    --judge-model "${VLLM_MODEL}" \
    --api --api-base "${VLLM_ENDPOINT}" \
    --grpo-group-size 4 \
    --grpo-max-iterations 10 \
    --grpo-num-bad-cases 20 \
    --grpo-case-chunk-size 5 \
    --grpo-early-stop-patience 5 \
    --grpo-early-stop-warmup 3 \
    --grpo-temperature 0.7 \
    --grpo-min-apply-threshold 0.2 \
    --grpo-max-parse-retries 2 \
    --reward-metric llm_judge \
    --device cpu \
    2>&1 | tee logs/grpo_local_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "[Done] GRPO Local Training End: $(date)"
echo "Check grpo_data/local/ for exported samples."