#!/bin/bash
# 极简验证方案：3 inner epochs × 1 outer epoch
# 目标: 验证训练循环是否能改善QA F1, 预计12.8小时完成
# 
# 精确估算:
#   每inner_epoch: 560 sessions × 2 calls + 1986 QAs × 0.3 = 1,120 + 596 = 1,716 calls
#   总计: 3 × 1,716 = 5,148 calls
#   时间: 5,148 × 9s = 46,332s ≈ 12.9h
#   tokenPlan成本: ¥0 (企业内部免费)
#
# 验证指标: 比较inner_epoch 1 vs 3的QA F1变化趋势

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache

# 确保SSH隧道存在 (API需要通过隧道访问)
if ! pgrep -f "ssh.*19443.*modelservice" > /dev/null 2>&1; then
    echo "[WARN] SSH tunnel for API not detected. Make sure tunnel is active:"
   echo "  ssh -R 19443:modelservice.jdcloud.com:443 nb-m7wkdajepx"
fi

echo "=== 极简验证训练: 3 inner × 1 outer ==="
echo "预计完成时间: ~13小时"
echo "Start: $(date)"
echo ""

/opt/conda/bin/python main.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --model maas-token-latest \
    --api --api-base "https://localhost:19443/tokenPlan/openai/v1" \
    --api-key "pk-8c8b5a47-95e1-4609-8272-35ccc50934f8" \
    --controller-type llm \
    --retriever /root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
    --state-encoder /root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
    --op-encoder /root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
    --inner-epochs 3 --outer-epochs 1 \
    --batch-size 1 \
    --encode-batch-size 16 \
    --session-mode full-session \
    --action-top-k 3 \
    --mem-top-k 20 --mem-top-k-eval 20 \
    --max-new-tokens 4096 \
    --reward-metric f1 \
    --locomo-train-query-sampling-ratio 0.3 \
    --device cpu \
    --disable-flash-attn \
    --wandb-run-name minimal_verify \
    --save-dir ./checkpoints/minimal_verify \
    --out-file ./results/minimal_verify.json \
    2>&1 | tee logs/minimal_verify_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "End: $(date)"
echo "=== Done ==="