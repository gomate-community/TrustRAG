# Use the official Ubuntu base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# ==========================================
# 第一阶段：系统级配置 (变动最少，放最前)
# ==========================================
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_NVML_BASED_CUDA_CHECK=1
ENV LANG C.UTF-8

# 1. 换源 (系统源)
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list \
    && sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

# 2. 安装系统依赖 (这些非常耗时，但几乎不需要改，所以放前面缓存起来)
# 合并 apt-get 指令以减少层数和体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    vim net-tools procps lsof curl wget iputils-ping telnet lrzsz git \
    libreoffice libmagic-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
    # rm -rf 是为了清理缓存减小体积

# ==========================================
# 第二阶段：Python 依赖 (变动偶尔)
# ==========================================
# 设置 pip 源
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 只有当 requirements.txt 变化时，才会触发下面的 pip install
COPY requirements.txt /workspace/requirements.txt
# 使用 --no-cache-dir 减小体积
RUN pip install -r /workspace/requirements.txt --no-cache-dir

# ==========================================
# 第三阶段：静态大文件 (变动偶尔)
# ==========================================
COPY nltk_data /root/nltk_data

# ==========================================
# 第四阶段：项目源代码 (变动最频繁，放最后)
# ==========================================
# 这样你改代码时，上面所有层都会直接用缓存，构建只需要 1 秒
COPY api /app/api
COPY trustrag /app/trustrag

# ==========================================
# 第五阶段：运行配置
# ==========================================
ENV PYTHONPATH=/app
WORKDIR /app/api/rag

CMD ["python", "main.py"]