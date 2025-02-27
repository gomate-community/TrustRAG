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
  --port 8000"
```

