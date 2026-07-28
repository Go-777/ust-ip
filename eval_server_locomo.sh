#!/bin/bash
# Server eval script for LoCoMo (via SSH tunnel on port 19443)
export WANDB_MODE=disabled

/opt/conda/bin/python main.py \
    --controller-type llm \
    --memory-cache-suffix "locomo_server_llm" \
    --eval-only \
    --inference-workers 1 \
    --inference-session-workers 1 \
    --action-top-k 3 \
    --mem-top-k-eval 5 \
    --session-mode turn-pair \
    --dataset locomo \
    --data-file ./data/locomo10.json \
    --model maas-token-latest \
    --api \
    --api-base "https://modelservice.jdcloud.com:19443/tokenPlan/openai/v1" \
    --api-key "pk-8c8b5a47-95e1-4609-8272-35ccc50934f8" \
    --selector-model maas-token-latest \
    --retriever qwen3-embedding-0.6b \
    --state-encoder sentence-transformers/all-MiniLM-L6-v2 \
    --disable-flash-attn \
    --enable-designer \
    --device cpu \
    --wandb-run-name eval_locomo_server \
    --save-dir ./checkpoints/server_llm \
    --out-file ./results/server_locomo.json