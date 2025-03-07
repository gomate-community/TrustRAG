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



## 推送阿里云

```bash
docker login --username=11859*****@qq.com registry.cn-beijing.aliyuncs.com
docker tag [ImageId] registry.cn-beijing.aliyuncs.com/quincyqiang/trustrag:[镜像版本号]
docker push registry.cn-beijing.aliyuncs.com/quincyqiang/trustrag:[镜像版本号]
```