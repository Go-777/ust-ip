#!/bin/bash
# ============================================================
# 一键运行: vLLM部署 + GRPO训练（JD Cloud H200）
#
# 用法:
#   ssh root@nb-m7wkdajepx
#   cd /mnt/workspace/home/zhaozhichen1/MemSkill
#   tmux new -s grpo_local
#   bash run_grpo_local.sh
# ============================================================

set -e

echo "====== Phase 1: Deploy vLLM (Qwen2.5-7B) ======"
bash scripts/deploy_vllm_local.sh

echo ""
echo "====== Phase 2: Sync data (if needed) ======"
# 确保 bad_cases_100.json 存在
if [ ! -f "./data/bad_cases_100.json" ]; then
    echo "ERROR: data/bad_cases_100.json not found!"
    echo "Please run on local machine first:"
    echo "  python scripts/generate_bad_cases.py --num-cases 100 --output data/bad_cases_100.json"
    echo "Then git push and pull on server."
    exit 1
fi
echo "  ✓ bad_cases_100.json exists ($(wc -l < ./data/bad_cases_100.json) lines)"

echo ""
echo "====== Phase 3: Run GRPO Training ======"
bash train_grpo_local.sh

echo ""
echo "====== All Done! ======"