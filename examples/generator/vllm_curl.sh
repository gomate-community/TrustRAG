curl http://localhost:8002/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3-32B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20
}'