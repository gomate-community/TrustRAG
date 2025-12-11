#!/usr/bin/env sh
set -eu

# Start the RAG API (api/rag/main.py) inside the trustrag:v0.1 image.
# POSIX 兼容，sh 或 bash 均可执行。
# 用法：
#   sh start_rag.sh
#   ENV_FILE=path/to/.env HOST_PORT=18000 IMAGE_NAME=trustrag:v0.1 sh start_rag.sh

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
ENV_FILE=${ENV_FILE:-"$ROOT_DIR/api/rag/.env"}
HOST_PORT=${HOST_PORT:-10000}
CONTAINER_PORT=10000
IMAGE_NAME=${IMAGE_NAME:-trustrag:v0.1}

if [ ! -f "$ENV_FILE" ]; then
  echo "[WARN] Env file not found at $ENV_FILE, continuing without --env-file"
  ENV_ARG=""
else
  ENV_ARG="--env-file $ENV_FILE"
fi

# 挂载整个仓库，确保能找到 trustrag 包
docker run --rm \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  $ENV_ARG \
  -v "$ROOT_DIR":/app \
  -w /app/api/rag \
  "$IMAGE_NAME" \
  sh -c "PYTHONPATH=/app python main.py"



