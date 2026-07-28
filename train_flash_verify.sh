#!/bin/bash
#SBATCH --job-name=memskill_flash_verify
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grpo_flash_verify_%j.out
#SBATCH --error=logs/grpo_flash_verify_%j.err

# ============================================================
# MemSkill GRPO 低成本验证 (qwen3.5-flash)
# ============================================================
# 目标: 用便宜模型跑3轮，验证reward信号改善后的效果:
#   1. reward分数是否有意义分化（好skill > 差skill）
#   2. executor output format是否规范（新增的FORMAT指导有效）
#   3. initial_memory_count传递链是否正常工作
#
# 模型: qwen3.5-flash (0.2元/M tokens, 比plus便宜8-40倍)
# 参数: 3轮迭代, group_size=4, 10条bad_cases
# 预计: 15-20分钟, 费用 <1元
# ============================================================

# === 环境配置 ===
eval "$(conda shell.bash hook)"
conda activate memskill

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

# 百炼API配置
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY='REDACTED_API_KEY'

# 项目目录
cd ~/ust-ip
mkdir -p logs checkpoints/grpo_flash_verify grpo_data/flash_verify

echo "============================================================"
echo "  MemSkill GRPO Flash Verify (qwen3.5-flash, low cost)"
echo "  Node: $(hostname)"
echo "  Start: $(date)"
echo "============================================================"

# === API连通性测试 ===
echo "[Check] API connectivity..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    "${DASHSCOPE_API_BASE}/models" \
    -H "Authorization: Bearer ${DASHSCOPE_API_KEY}" 2>/dev/null)
echo "  HTTP code: ${HTTP_CODE}"
if [ "$HTTP_CODE" = "000" ]; then
    echo "  [FATAL] Cannot reach dashscope API!"
    exit 1
fi

# === GRPO低成本验证: 3轮, group_size=4, 10条case ===
echo ""
echo "[Run] GRPO flash verify: 3 iters, group_size=4, 10 bad cases..."
echo "  Model: qwen3.5-flash (~0.2 yuan/M tokens)"
echo "  Expected cost: <1 yuan"
echo ""
python train_grpo.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen3.5-flash \
    --api --api-base "${DASHSCOPE_API_BASE}" \
    --grpo-enabled \
    --grpo-group-size 4 \
    --grpo-max-iterations 3 \
    --grpo-num-bad-cases 10 \
    --grpo-case-chunk-size 5 \
    --grpo-temperature 0.7 \
    --grpo-max-designer-tokens 4096 \
    --grpo-early-stop-patience 5 \
    --action-top-k 3 \
    --max-new-tokens 4096 \
    --device cuda \
    --save-dir ./checkpoints/grpo_flash_verify \
    --grpo-export-dir ./grpo_data/flash_verify

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "  Exit code: ${EXIT_CODE}"
echo "  End: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[Results] Checking reward signal quality..."
    echo "Look for: reward variance > 0 (differentiated scores)"
    echo ""
    # 显示训练输出中的reward统计
    if ls grpo_data/flash_verify/*.jsonl 1> /dev/null 2>&1; then
        echo "GRPO export files:"
        ls -la grpo_data/flash_verify/*.jsonl
        echo ""
        echo "Sample entries (first 3):"
        head -n 3 grpo_data/flash_verify/*.jsonl
    fi
fi