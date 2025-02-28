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


## 参数说明

### Docker 运行参数

| 参数 | 值 | 说明 |
|-----------|-------|-------------|
| `-d` | - | 以分离模式运行容器（在后台运行） |
| `-p` | `3000:8080` | 将主机上的3000端口映射到容器内的8080端口 |
| `--name` | `open-webui` | 为容器分配一个名称 |
| `--restart` | `always` | 容器停止时总是自动重启 |

### 环境变量

| 变量 | 值 | 说明 |
|----------|-------|-------------|
| `OPENAI_API_KEY` | `your_secret_key` | OpenAI API的认证密钥 |
| `ENABLE_OLLAMA_API` | `False` | 禁用Ollama API集成 |
| `ENABLE_OPENAI_API` | `True` | 启用OpenAI API集成 |
| `OPENAI_API_BASE_URL` | `http://192.168.1.5:8000/v1` | 自定义OpenAI API端点URL |
| `DEFAULT_MODELS` | `"DeepSeek-R1-Distill-Qwen-1.5B"` | 要使用的默认模型 |
| `RAG_EMBEDDING_MODEL` | `"/app/all-MiniLM-L6-v2"` | RAG（检索增强生成）使用的嵌入模型路径 |

### 卷挂载

| 源路径 | 目标路径 | 说明 |
|-------------|-------------|-------------|
| `/mnt/g/Ubuntu_WSL/open-webui` | `/app/backend/data` | Open WebUI数据的持久存储 |
| `/mnt/g/pretrained_models/mteb/all-MiniLM-L6-v2` | `/app/all-MiniLM-L6-v2` | 将本地嵌入模型挂载到容器中 |

### 镜像

| 参数 | 值 | 说明 |
|-----------|-------|-------------|
| 镜像 | `ghcr.io/open-webui/open-webui:main` | GitHub容器注册表中的Open WebUI镜像（main分支） |

## 注意事项

- 此配置设置为使用运行在`http://192.168.1.5:8000/v1`的OpenAI兼容API
- 使用本地嵌入模型而不是从互联网下载
- 通过卷映射启用数据持久化
- 主机上的3000端口将用于访问UI界面