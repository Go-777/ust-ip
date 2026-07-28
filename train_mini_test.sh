#!/bin/bash
# Mini训练测试：跑3个inner_epoch估算API消耗
# 预期调用: 3 epochs × (4 episodes × 19 sessions × 2 calls + 4 × 200 QA) = 3 × (152 + 800) = 2,856 calls
# 但实际会因为QA sampling ratio=0.3而减少到约: 3 × (152 + 240) = 1,176 calls

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0

echo "=== Mini Train Test: 3 inner epochs, estimate API cost ==="
echo "Start time: $(date)"

python main.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --model maas-token-latest \
    --api --api-base "https://modelservice.jdcloud.com/tokenPlan/openai/v1" \
    --api-key "pk-8c8b5a47-95e1-4609-8272-35ccc50934f8" \
    --controller-type llm \
    --retriever qwen3-embedding-0.6b \
    --inner-epochs 3 --outer-epochs 1 \
    --batch-size 1 \
    --encode-batch-size 4 \
    --session-mode full-session \
    --action-top-k 3 \
    --mem-top-k 20 --mem-top-k-eval 20 \
    --reward-metric f1 \
    --locomo-train-query-sampling-ratio 0.3 \
    --enable-designer \
    --designer-freq 1 \
    --device cpu \
    --wandb-run-name mini_train_test \
    --save-dir ./checkpoints/mini_test \
    --out-file ./results/mini_test.json \
    2>&1 | tee logs/mini_train_test_$(date +%Y%m%d_%H%M%S).log

echo "End time: $(date)"