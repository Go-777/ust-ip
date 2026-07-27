#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# LLM Controller mode: no PPO checkpoint needed
python main.py \
    --controller-type llm \
    --memory-cache-suffix "hp_eval_llm" \
    --eval-only \
    --inference-workers 4 \
    --inference-session-workers 1 \
    --action-top-k 7 \
    --mem-top-k-eval 20 \
    --session-mode fixed-length \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --dataset hotpotqa \
    --hotpotqa-eval-file "[YOUR_EVAL_FILE_PATH]" \
    --data-file "[YOUR_DATA_PATH]" \
    --model maas-token-latest \
    --api \
    --api-base "https://modelservice.jdcloud.com/tokenPlan/openai/v1" \
    --api-key "pk-8c8b5a47-95e1-4609-8272-35ccc50934f8" \
    --selector-model maas-token-latest \
    --retriever qwen3-embedding-0.6b \
    --designer-freq 1 \
    --encode-batch-size 4 \
    --mem-top-k 20 \
    --reward-metric llm_judge \
    --device cuda \
    --enable-designer \
    --skip-load-snapshot-manager \
    --wandb-run-name eval_hp_llm \
    --save-dir ./checkpoints/hp_llm_controller \
    --out-file ./results/hp_llm_controller.json