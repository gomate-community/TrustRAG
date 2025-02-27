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

