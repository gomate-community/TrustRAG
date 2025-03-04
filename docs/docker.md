##  选择pytorch镜像

我们可以从dockerhub搜索合适的pytorch镜像，下面为了兼容更多功能，直接选用最新的tag
```bash
docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
```

## 构建镜像

```bash
docker build -t trustrag:0.1 .
```

```bash
docker build -f ./Dockerfile -t trustrag:1.0 .
```

## 启动容器

```bash
docker run -itd --gpus=all --name=test trustrag:0.1 /bin/bash
```

## 删除容器

```bash
docker rm -f test
```

## 启动vllm服务

- 推理内容
> 输出中包括content和reason_content两个字段
```bash
docker run -itd \
  --gpus=all \
  --name=llm_server \
  -v /mnt/g/pretrained_models/llm/DeepSeek-R1-Distill-Qwen-1.5B:/workspace/DeepSeek-R1-Distill-Qwen-1.5B \
  -p 8000:8000 \
  trustrag:0.1 \
  bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /workspace/DeepSeek-R1-Distill-Qwen-1.5B \
  --served-model-name DeepSeek-R1-Distill-Qwen-1.5B \
  --max-model-len=8192 \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size=1 \
  --swap-space=4 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --host 0.0.0.0 \
  --port 8000"
```

- think标签

>部署r1相关模型的时候，发现没有输出前半部分`<think>`标签，只输出了后半部分`</think>`，解决方案参考：https://github.com/deepseek-ai/DeepSeek-R1/issues/352

```bash
docker run -itd \
  --gpus=all \
  --name=llm_server \
  -v /mnt/g/pretrained_models/llm/DeepSeek-R1-Distill-Qwen-1.5B:/workspace/DeepSeek-R1-Distill-Qwen-1.5B \
  -p 8000:8000 \
  trustrag:0.1 \
  bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model /workspace/DeepSeek-R1-Distill-Qwen-1.5B \
  --served-model-name DeepSeek-R1-Distill-Qwen-1.5B \
  --max-model-len=8192 \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size=1 \
  --swap-space=4 \
  --host 0.0.0.0 \
  --port 8000 \
  --chat-template /mnt/g/pretrained_models/llm/DeepSeek-R1-Distill-Qwen-1.5B/template_deepseek_r1.jinja"
```


### Docker参数说明：

- `--gpus=all` - 允许容器访问主机上的所有GPU资源
- `--name=llm_server` - 将容器命名为"llm_server"，便于后续管理
- `-v /mnt/g/pretrained_models/llm/DeepSeek-R1-Distill-Qwen-1.5B:/workspace/DeepSeek-R1-Distill-Qwen-1.5B` - 将主机上的模型目录挂载到容器内的工作目录
- `-p 8000:8000` - 将容器内的8000端口映射到主机的8000端口
- `trustrag:0.1` - 使用名为"trustrag"，标签为"0.1"的Docker镜像

### VLLM服务参数说明：

- `CUDA_VISIBLE_DEVICES=0,1,2,3` - 指定使用的GPU设备编号
- `--model /workspace/DeepSeek-R1-Distill-Qwen-1.5B` - 指定模型路径
- `--served-model-name DeepSeek-R1-Distill-Qwen-1.5B` - 设置对外提供服务的模型名称
- `--max-model-len=8192` - 设置模型的最大序列长度为8192
- `--gpu-memory-utilization=0.9` - 设置GPU内存利用率为90%
- `--tensor-parallel-size=1` - 设置模型并行度为1（使用1个GPU处理模型）
- `--swap-space=4` - 设置交换空间大小为4GB
- `--enable-reasoning` - 启用推理功能（注意：此参数重复了两次）
- `--reasoning-parser deepseek_r1` - 指定使用deepseek_r1作为推理解析器
- `--host 0.0.0.0` - 监听所有网络接口
- `--port 8000` - 监听端口号为8000