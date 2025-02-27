# Use the official Ubuntu base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_NVML_BASED_CUDA_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8


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

# Set environment variables to non-interactive to avoid prompts during installation
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY requirements.txt /workspace
RUN pip install -r requirements.txt --no-cache
