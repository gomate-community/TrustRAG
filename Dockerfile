# Use the official Ubuntu base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_NVML_BASED_CUDA_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Set environment variables to non-interactive to avoid prompts during installation
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY requirements.txt /workspace
RUN pip install -r requirements.txt --no-cache

# 使用阿里云镜像源加速apt-get
#RUN sed -i 's@/archive.ubuntu.com/@/mirrors.aliyun.com/@g' /etc/apt/sources.list
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
#COPY sources.list /etc/apt/sources.list
#ADD sources.list /etc/apt
RUN apt-get clean

# 安装常用依赖包
RUN apt-get -q update \
    && apt-get -q install -y --no-install-recommends \
        apt-utils \
        bats \
        build-essential
RUN apt-get update && apt-get install -y vim net-tools procps lsof curl wget iputils-ping telnet lrzsz git libreoffice libmagic-dev

RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get gcc --version
RUN apt-get autoclean
RUN rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# 1. 补充缺失的静态资源
# ----------------------------------------------------
COPY nltk_data /root/nltk_data

# ----------------------------------------------------
# 2. 精确复制源代码 (拒绝复制乱七八糟的文件)
# ----------------------------------------------------
# 将本地的 api 目录复制到容器的 /app/api
COPY api /app/api
# 将本地的 trustrag 目录复制到容器的 /app/trustrag
COPY trustrag /app/trustrag

# ----------------------------------------------------
# 3. 固化运行环境设置 (替代命令行参数)
# ----------------------------------------------------
# 设置环境变量 (替代 sh -c "PYTHONPATH=/app ...")
ENV PYTHONPATH=/app

# 设置工作目录 (替代 -w /app/api/rag)
# 之后的所有 CMD 或 RUN 都会在这个目录下执行
WORKDIR /app/api/rag

# ----------------------------------------------------
# 4. 设置默认启动命令
# ----------------------------------------------------
# 容器启动时默认执行该命令
CMD ["python", "main.py"]