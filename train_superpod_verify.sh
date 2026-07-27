#!/bin/bash
#SBATCH --job-name=memskill_verify
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grpo_verify_%j.out
#SBATCH --error=logs/grpo_verify_%j.err

# ============================================================
# MemSkill GRPO 快速验证 (SuperPod)
# ============================================================
# 目标: 用最小参数跑通全链路，验证:
#   1. conda环境正常 (torch/vllm/trl)
#   2. API连通 (百炼dashscope)
#   3. GRPO pipeline端到端无报错
#   4. 输出JSONL文件格式正确
#
# 参数: 1次迭代, group_size=2, 2条bad_cases
# 预计: 10-15分钟完成, 消耗 <1 GPU-hour
# ============================================================

# === 环境配置 ===
# === 环境配置 ===
eval "$(conda shell.bash hook)"
conda activate memskill

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

# 百炼API配置 (通过环境变量传递，避免shell特殊字符问题)
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY='REDACTED_API_KEY'

# 项目目录
cd ~/ust-ip
mkdir -p logs checkpoints/grpo_verify grpo_data/verify

echo "============================================================"
echo "  MemSkill GRPO Quick Verify"
echo "  Node: $(hostname)"
echo "  Start: $(date)"
echo "============================================================"

# === 环境检查 ===
echo "[Check] Python env..."
python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')"
python -c "import transformers, trl; print(f'  transformers={transformers.__version__}, trl={trl.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# === API连通性测试 ===
echo "[Check] API connectivity..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    "${DASHSCOPE_API_BASE}/models" \
    -H "Authorization: Bearer ${DASHSCOPE_API_KEY}" 2>/dev/null)
echo "  HTTP code: ${HTTP_CODE}"
if [ "$HTTP_CODE" = "000" ]; then
    echo "  [FATAL] Cannot reach dashscope API from compute node!"
    echo "  Try: curl -v ${DASHSCOPE_API_BASE}/models"
    exit 1
fi

# === GRPO最小验证 ===
echo ""
echo "[Run] GRPO mini verify: 1 iter, group_size=2, 2 bad cases..."
python train_grpo.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --bad-cases-file "./data/bad_cases_mini.json" \
    --model qwen-plus \
    --api --api-base "${DASHSCOPE_API_BASE}" \
    --grpo-enabled \
    --grpo-group-size 2 \
    --grpo-max-iterations 1 \
    --grpo-num-bad-cases 2 \
    --grpo-case-chunk-size 2 \
    --grpo-temperature 0.7 \
    --grpo-max-designer-tokens 2048 \
    --action-top-k 2 \
    --max-new-tokens 2048 \
    --device cuda \
    --save-dir ./checkpoints/grpo_verify \
    --grpo-export-dir ./grpo_data/verify

echo ""
echo "[Done] End: $(date)"
echo "============================================================"