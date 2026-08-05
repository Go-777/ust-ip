#!/bin/bash
# ============================================================
# vLLM 本地部署脚本 — JD Cloud H200
# 部署 Qwen2.5-7B-Instruct 用于 GRPO 验证
# 
# 用法: 
#   ssh root@nb-m7wkdajepx
#   cd /mnt/workspace/home/zhaozhichen1/MemSkill
#   bash scripts/deploy_vllm_local.sh
#
# 部署后 endpoint: http://localhost:8000/v1
# 模型名: Qwen/Qwen2.5-7B-Instruct
# ============================================================

set -e

# === 代理配置（JD Cloud 外网必须） ===
export http_proxy=http://bamboo-proxy.jd.com:80
export https_proxy=http://bamboo-proxy.jd.com:80

# === 环境 ===
export PATH="/mnt/workspace/envs/vllm-py312/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR="/mnt/workspace/home/zhaozhichen1/models/Qwen2.5-7B-Instruct"
PORT=8000

echo "============================================"
echo "  vLLM Local Deployment"
echo "  Model: ${MODEL_NAME}"
echo "  Port: ${PORT}"
echo "  Start: $(date)"
echo "============================================"

# === Step 1: 下载模型（如果不存在） ===
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[Step 1] Downloading model from ModelScope..."
    pip install modelscope -q 2>/dev/null || true
    python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', 
                  cache_dir='/mnt/workspace/home/zhaozhichen1/models',
                  local_dir='${MODEL_DIR}')
print('Download complete!')
"
else
    echo "[Step 1] Model already exists at ${MODEL_DIR}"
fi

# === Step 2: 启动 vLLM 服务 ===
echo ""
echo "[Step 2] Starting vLLM server..."
echo "  Endpoint: http://localhost:${PORT}/v1"
echo "  Model ID: ${MODEL_NAME}"
echo ""

# 先清理可能存在的旧进程
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# 关闭代理（vLLM服务本身不需要外网）
unset http_proxy https_proxy

# 启动 vLLM（后台运行）
nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --dtype auto \
    --trust-remote-code \
    > logs/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"

# === Step 3: 等待服务就绪 ===
echo ""
echo "[Step 3] Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "  ✓ vLLM is ready! (took ${i}s)"
        echo ""
        echo "============================================"
        echo "  vLLM deployed successfully!"
        echo "  Endpoint: http://localhost:${PORT}/v1"
        echo "  Model: ${MODEL_NAME}"
        echo "  PID: ${VLLM_PID}"
        echo "  Log: logs/vllm_server.log"
        echo "============================================"
        echo ""
        # 验证一下
        curl -s http://localhost:${PORT}/v1/models | python -m json.tool
        exit 0
    fi
    sleep 2
done

echo "  ✗ vLLM failed to start within 120s"
echo "  Check logs: tail -50 logs/vllm_server.log"
exit 1