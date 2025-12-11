# 使用默认 .env，再覆盖 RERANKER_NAME
sh start_rag.sh -e RERANKER_NAME=G:/pretrained_models/mteb/bge-reranker-large-v2


docker run --rm \
  --env-file /path/to/.env \
  -e RERANKER_NAME=G:/pretrained_models/mteb/bge-reranker-large-v2 \
  -p 10000:10000 \
  -v /g/Projects/TrustRAG:/app \
  -w /app/api/rag \
  trustrag:v0.1 \
  sh -c "PYTHONPATH=/app python main.py"



docker run --rm \
  --gpus all \
  -p 10000:10000 \
  -v /mnt/g/pretrained_models/mteb/bge-reranker-large:/mnt/g/pretrained_models/mteb/bge-reranker-large \
  -v .:/app \
  -e RERANKER_NAME=/mnt/g/pretrained_models/mteb/bge-reranker-large \
  -w /app/api/rag \
  trustrag:v0.1 \
  sh -c "PYTHONPATH=/app python main.py"
