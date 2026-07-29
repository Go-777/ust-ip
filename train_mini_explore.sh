#!/bin/bash
# ============================================================
# MemSkill GRPO 小批量完整探索实验
# ============================================================
# 运行方式 (不怕熄屏中断):
#   SuperPod: sbatch train_mini_explore.sh
#   本地Mac: nohup bash train_mini_explore.sh &
#   查看进度: tail -f logs/grpo_mini_explore_*.log
#
# 预计运行时间: 10-15分钟 (~60次API调用, qwen3-max)
# ============================================================
# 目标: 跑完一段完整的多轮迭代轨迹，观察:
#   1. reward是否能从0开始出现分化
#   2. 迭代间reward是否有上升趋势
#   3. 最终生成的skill改进是否合理
#
# 参数: 2轮, group_size=2, 3 bad_cases, chunk_size=3
# 模型: qwen3-max (主pipeline + judge)
# 预计耗时: 10-15分钟 (~60次API调用)
# 预计消耗: ~50K tokens
# ============================================================

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

# 百炼API配置
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Error: DASHSCOPE_API_KEY not set}"

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./checkpoints/grpo_mini_explore"
EXPORT_DIR="./grpo_data/mini_explore"
LOG_FILE="logs/grpo_mini_explore_${TIMESTAMP}.log"

mkdir -p logs "${SAVE_DIR}" "${EXPORT_DIR}"

echo "============================================================" | tee "${LOG_FILE}"
echo "  MemSkill GRPO Mini Explore (qwen3-max)" | tee -a "${LOG_FILE}"
echo "  Start: $(date)" | tee -a "${LOG_FILE}"
echo "  Parameters:" | tee -a "${LOG_FILE}"
echo "    - iterations: 5 (enough to see trend)" | tee -a "${LOG_FILE}"
echo "    - group_size: 3 (min for meaningful comparison)" | tee -a "${LOG_FILE}"
echo "    - bad_cases: 6 (small but diverse)" | tee -a "${LOG_FILE}"
echo "    - chunk_size: 3 (2 chunks per iteration)" | tee -a "${LOG_FILE}"
echo "    - reward: llm_judge (qwen3-max, strongest signal)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

python train_grpo.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen3.7-flash \
    --selector-model qwen3.7-flash-2026-07-15 \
    --executor-model qwen3.5-plus-2026-04-20 \
    --designer-local-model qwen3.7-plus-2026-05-26 \
    --judge-model qwen3.7-max-2026-06-08 \
    --api --api-base "${DASHSCOPE_API_BASE}" \
    --api-key "${DASHSCOPE_API_KEY}" \
    --grpo-enabled \
    --grpo-group-size 4 \
    --grpo-max-iterations 3 \
    --grpo-num-bad-cases 6 \
    --grpo-case-chunk-size 3 \
    --grpo-temperature 0.9 \
    --grpo-max-designer-tokens 2048 \
    --grpo-early-stop-patience 5 \
    --action-top-k 2 \
    --max-new-tokens 2048 \
    --reward-metric llm_judge \
    --device cpu \
    --save-dir "${SAVE_DIR}" \
    --grpo-export-dir "${EXPORT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "  Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "  End: $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "[Results Summary]" | tee -a "${LOG_FILE}"
    echo "  Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "  Checkpoint: ${SAVE_DIR}/" | tee -a "${LOG_FILE}"
    echo "  GRPO export: ${EXPORT_DIR}/" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "Key metrics to check in log:" | tee -a "${LOG_FILE}"
    echo "  1. grep 'Avg reward' ${LOG_FILE}  → reward趋势" | tee -a "${LOG_FILE}"
    echo "  2. grep 'REWARD-DEBUG' ${LOG_FILE} → 每步细节" | tee -a "${LOG_FILE}"
    echo "  3. grep 'improvement' ${LOG_FILE}  → 是否有提升" | tee -a "${LOG_FILE}"

    if ls ${EXPORT_DIR}/*.jsonl 1> /dev/null 2>&1; then
        echo "" | tee -a "${LOG_FILE}"
        echo "GRPO export files:" | tee -a "${LOG_FILE}"
        ls -la ${EXPORT_DIR}/*.jsonl | tee -a "${LOG_FILE}"
    fi
fi