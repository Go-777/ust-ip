#!/bin/bash
#SBATCH --job-name=memskill_grpo
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grpo_train_%j.out
#SBATCH --error=logs/grpo_train_%j.err

# ============================================================
# MemSkill GRPO Training on HKUST SuperPod (H800 80GB)
# ============================================================
# 训练流程:
#   1. 从bad_cases中分析失败模式 (API调用)
#   2. Designer生成G个候选skill方案 (API调用)
#   3. 对每个候选方案: apply skill → re-run bad cases → compute reward
#   4. 导出GRPO训练数据 (JSONL格式, 供OpenRLHF微调)
#
# 资源预估:
#   - GPU: 1×H800 80GB (vLLM推理 + 后续weight update)
#   - API调用: 百炼 qwen-plus (~50次迭代 × 8组 = ~400次)
#   - 时间: ~4-6小时 (视API延迟)
#   - GPU小时: 1×6h = 6 GPU-hours
# ============================================================

# === 环境配置 ===
source ~/.bashrc
conda activate memskill

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

# 百炼API配置
DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY="REDACTED_API_KEY"

# 项目目录
cd ~/MemSkill
mkdir -p logs checkpoints/grpo_superpod grpo_data/superpod

echo "============================================================"
echo "  MemSkill GRPO Training"
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start: $(date)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'runtime only')"
echo "============================================================"

# === Step 1: 验证API连通性 ===
echo "[Step 1] Testing API connectivity..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    "${DASHSCOPE_API_BASE}/models" \
    -H "Authorization: Bearer ${DASHSCOPE_API_KEY}" 2>/dev/null)
echo "  API response code: ${HTTP_CODE}"
if [ "$HTTP_CODE" = "000" ]; then
    echo "  [ERROR] API not reachable! Check network."
    exit 1
fi

# === Step 2: GRPO训练 (LoCoMo数据集) ===
echo ""
echo "[Step 2] Starting GRPO training on LoCoMo bad cases..."
python train_grpo.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --bad-cases-file "./data/bad_cases_extended.json" \
    --model qwen-plus \
    --api --api-base "${DASHSCOPE_API_BASE}" \
    --api-key "${DASHSCOPE_API_KEY}" \
    --grpo-enabled \
    --grpo-group-size 8 \
    --grpo-max-iterations 50 \
    --grpo-num-bad-cases 100 \
    --grpo-case-chunk-size 5 \
    --grpo-temperature 0.7 \
    --grpo-max-designer-tokens 4096 \
    --grpo-early-stop-patience 5 \
    --action-top-k 3 \
    --max-new-tokens 4096 \
    --device cuda \
    --save-dir ./checkpoints/grpo_superpod \
    --grpo-export-dir ./grpo_data/superpod

echo ""
echo "  End: $(date)"
echo "============================================================"