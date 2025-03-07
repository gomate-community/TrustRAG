## ollama部署

### docker部署

默认情况下，不同操作系统存储的路径如下：
```text
macOS: ~/.ollama/models
Linux: /usr/share/ollama/.ollama/models
Windows: C:\Users<username>.ollama\models
```

export OLLAMA_MODELS=/data/ollama/models

### 下载模型

在这里查看模型：https://ollama.com/search
```bash
ollama pull deepseek-r1:1.5b
```

### 导入本地模型


[导入本地模型](https://github.com/ollama/ollama/blob/main/docs/docker.md)

```bash
docker run -d -p 11434:11434 \
  --gpus=all \
  --name ollama \
  -v /mnt/g/pretrained_models/llm:/root/pretrained_models/llm \
  ollama/ollama:latest
```

创建Modelfile
```text
FROM /root/pretrained_models/llm/DeepSeek-R1-Distill-Qwen-1.5B 
```

在Modelfile所在目录执行命令：
```bash
ollama create deepseek-r1:1.5b
```
运行模型

```bash
ollama run deepseek-r1:1.5b
```

列举模型

```bash
ollama list
```

查看模型

```bash
ollama show deepseek-r1:1.5b
```

删除模型
```bash
ollama rm deepseek-r1:1.5b
```



### api服务

参考文档：https://github.com/ollama/ollama/blob/main/docs/api.md
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1:1.5b",
  "prompt": "怎么学习机器学习?",
  "stream": false
}'
```
## 参考资料：

[Ollama完整教程：本地LLM管理、WebUI对话、Python/Java客户端API应用](https://www.cnblogs.com/obullxl/p/18295202/NTopic2024071001)