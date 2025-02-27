## 启动open-webui
```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_KEY=your_secret_key \
  -e ENABLE_OLLAMA_API=False \
  -e ENABLE_OPENAI_API=True \
  -e OPENAI_API_BASE_URL=http://192.168.1.5:8000/v1 \
  -e DEFAULT_MODELS="DeepSeek-R1-Distill-Qwen-1.5B" \
  -e RAG_EMBEDDING_MODEL="/app/all-MiniLM-L6-v2" \
  -v /mnt/g/Ubuntu_WSL/open-webui:/app/backend/data \
  -v /mnt/g/pretrained_models/mteb/all-MiniLM-L6-v2:/app/all-MiniLM-L6-v2 \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```