## 环境依赖

- 安装 Docker

## 安装 Milvus

Milvus 在 Milvus 存储库中提供了 Docker Compose 配置文件。要使用 Docker Compose 安装 Milvus，只需运行

```bash
# Download the configuration file
$ wget https://github.com/milvus-io/milvus/releases/download/v2.5.3/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
$ sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```
```yml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.5.3
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

启动 Milvus 后， 名为milvus-standalone、milvus-minio和milvus-etcd的容器已启动。
- milvus-etcd容器不向主机暴露任何端口，并将其数据映射到当前文件夹中的volumes/etcd 。
- milvus-minio容器使用默认身份验证凭据在本地服务端口9090和9091 ，并将其数据映射到当前文件夹中的volumes/minio 。
- milvus-standalone容器使用默认设置在本地服务端口19530 ，并将其数据映射到当前文件夹中的volumes/milvus

其中minio访问地址：`http://localhost:9001/browser`,登录密码和用户名均为minioadmin

![](https://i-blog.csdnimg.cn/direct/12b0ebb2880b45e8b0d2fd384eed8c34.png)


## 停止并删除 Milvus
```bash
sudo docker compose down
sudo rm -rf volumes
```
可以按如下方式停止并删除该容器


## 安装图形化管理工具Attu

以下是整理后的 Markdown 文档：

```markdown
# 从 Docker 运行 Attu

## 启动容器运行 Attu 的步骤

```bash
docker run -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:v2.4
```

确保 Attu 容器可以访问 Milvus IP 地址。启动容器后，打开 Web 浏览器并输入 `http://{ Attu IP }:8000` 以查看 Attu GUI。

### 运行 Attu Docker 的可选环境变量

| 范围               | 例子                     | 必需的 | 描述                                      |
|--------------------|--------------------------|--------|-------------------------------------------|
| MILVUS_URL         | 192.168.0.1:19530        | 否     | 可选，Milvus 服务器 URL                   |
| 数据库             | 你的数据库               | 否     | 可选，默认数据库名称                      |
| ATTU_LOG_LEVEL     | 信息                     | 否     | 可选，设置 Attu 的日志级别                |
| 根证书路径         | /路径/到/根/证书         | 否     | 可选，根证书的路径                        |
| PRIVATE_KEY_PATH   | /路径/到/私人/密钥       | 否     | 可选，私钥路径                            |
| CERT_CHAIN_PATH    | /路径/到/证书/链         | 否     | 可选，证书链的路径                        |
| 服务器名称         | 你的服务器名称           | 否     | 可选，您的服务器名称                      |
| 服务器端口         | 服务器监听端口           | 否     | 可选，若未设置则默认为 3000               |

**请注意，`MILVUS_URL` 地址必须是 Attu Docker 容器可以访问的地址。因此，“127.0.0.1”或“localhost”不起作用。**

### 使用环境变量运行 Docker 容器

### Attu SSL 示例

```bash
docker run -p 8000:3000 \
-v /your-tls-file-path:/app/tls \
-e ATTU_LOG_LEVEL=info  \
-e ROOT_CERT_PATH=/app/tls/ca.pem \
-e PRIVATE_KEY_PATH=/app/tls/client.key \
-e CERT_CHAIN_PATH=/app/tls/client.pem \
-e SERVER_NAME=your_server_name \
zilliz/attu:dev
```

### 自定义服务器端口示例

此命令允许您使用主机网络运行 docker 容器，并为服务器指定要侦听的自定义端口。

```bash
docker run --network host \
-v /your-tls-file-path:/app/tls \
-e ATTU_LOG_LEVEL=info  \
-e SERVER_NAME=your_server_name \
-e SERVER_PORT=8080 \
zilliz/attu:dev
```
安装访问：`http://localhost:8000/#/connect`
![](https://i-blog.csdnimg.cn/direct/04b39b59371c46eb8a5421e5a2cde4e8.png)
![](https://i-blog.csdnimg.cn/direct/37055d01a73549b7b1706e8679ba64d3.png)

## 使用pymilvus操作Milvus
安装依赖环境：
```bash
pip install --upgrade pymilvus openai requests tqdm
```


以下是您提供的代码和说明的Markdown格式版本：

```markdown
# 准备数据

我们使用Milvus文档2.4.x中的常见问题解答页面作为我们RAG中的私有知识，这对于简单的RAG管道来说是一个很好的数据源。

下载zip文件并将文档提取到文件夹`milvus_docs`中。

```bash
$ wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
$ unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
```

我们从文件夹`milvus_docs/en/faq`中加载所有的markdown文件。对于每个文档，我们简单地用“#”来分隔文件中的内容，这样可以粗略地区分markdown文件各个主体部分的内容。

```python
from glob import glob

text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")
```

### 准备嵌入模型

我们初始化OpenAI客户端来准备嵌入模型。

```python
from openai import OpenAI

openai_client = OpenAI()
```

定义一个函数，使用OpenAI客户端生成文本嵌入。我们使用`text-embedding-3-small`模型作为示例。

```python
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
```

生成测试嵌入并打印其维度和前几个元素。

```python
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])
```

输出：

```
1536
[0.00988506618887186, -0.005540902726352215, 0.0068014683201909065, -0.03810417652130127, -0.018254263326525688, -0.041231658309698105, -0.007651153020560741, 0.03220026567578316, 0.01892443746328354, 0.00010708322952268645]
```


###  创建集合

```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./milvus_demo.db")

collection_name = "my_rag_collection"
```

至于`MilvusClient`的参数：

- 将`uri`设置为本地文件（例如`./milvus.db`）是最方便的方法，因为它会自动利用Milvus Lite将所有数据存储在此文件中。
- 如果你有大量数据，你可以在Docker或Kubernetes上搭建性能更佳的Milvus服务器。在此设置中，请使用服务器uri，例如`http://localhost:19530`，作为你的`uri`。
- 如果您想使用Milvus的完全托管云服务Zilliz Cloud，请调整`uri`和`token`，它们对应于Zilliz Cloud中的公共端点和Api密钥。

检查该集合是否已存在，如果存在则将其删除。

```python
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
```

使用指定的参数创建一个新的集合。

如果我们不指定任何字段信息，Milvus会自动创建一个默认`id`字段作为主键，以及一个`vector`字段用于存储向量数据。保留的JSON字段用于存储非架构定义的字段及其值。

```python
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)
```

###  将数据加载到Milvus中


遍历文本行，创建嵌入，然后将数据插入Milvus。

这里新增了一个字段`text`，是集合架构中未定义的字段，它将被自动添加到保留的JSON动态字段中，在高层次上可以将其视为普通字段。

```python
from tqdm import tqdm

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
```

输出：

```
Creating embeddings: 100%|██████████| 72/72 [00:27<00:00,  2.67it/s]

{'insert_count': 72,
 'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
 'cost': 0}
```


### 检索查询数据

让我们指定一个有关Milvus的常见问题。

```python
question = "How is data stored in milvus?"
```

在集合中搜索问题并检索语义前3个匹配。

```python
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # 使用`emb_text`函数将问题转换为嵌入向量
    limit=3,  # 返回前3个结果
    search_params={"metric_type": "IP", "params": {}},  # 内积距离
    output_fields=["text"],  # 返回text字段
)
```

让我们看一下查询的搜索结果。

```python
import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
```

输出：

```json
[
    [
        " Where does Milvus store data?\n\nMilvus deals with two types of data, inserted data and metadata. \n\nInserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends, including [MinIO](https://min.io/), [AWS S3](https://aws.amazon.com/s3/?nc1=h_ls), [Google Cloud Storage](https://cloud.google.com/storage?hl=en#object-storage-for-companies-of-all-sizes) (GCS), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs), [Alibaba Cloud OSS](https://www.alibabacloud.com/product/object-storage-service), and [Tencent Cloud Object Storage](https://www.tencentcloud.com/products/cos) (COS).\n\nMetadata are generated within Milvus. Each Milvus module has its own metadata that are stored in etcd.\n\n###",
        0.7883545756340027
    ],
    [
        "How does Milvus handle vector data types and precision?\n\nMilvus supports Binary, Float32, Float16, and BFloat16 vector types.\n\n- Binary vectors: Store binary data as sequences of 0s and 1s, used in image processing and information retrieval.\n- Float32 vectors: Default storage with a precision of about 7 decimal digits. Even Float64 values are stored with Float32 precision, leading to potential precision loss upon retrieval.\n- Float16 and BFloat16 vectors: Offer reduced precision and memory usage. Float16 is suitable for applications with limited bandwidth and storage, while BFloat16 balances range and efficiency, commonly used in deep learning to reduce computational requirements without significantly impacting accuracy.\n\n###",
        0.6757288575172424
    ],
    [
        "How much does Milvus cost?\n\nMilvus is a 100% free open-source project.\n\nPlease adhere to [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) when using Milvus for production or distribution purposes.\n\nZilliz, the company behind Milvus, also offers a fully managed cloud version of the platform for those that don't want to build and maintain their own distributed instance. [Zilliz Cloud](https://zilliz.com/cloud) automatically maintains data reliability and allows users to pay only for what they use.\n\n###",
        0.6421123147010803
    ]
]
```

### 使用LLM获取RAG响应

将检索到的文档转换为字符串格式。

```python
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
```
## 参考资料

- 安装:[使用 Docker Compose 运行 Milvus（Linux）](https://milvus.io/docs/install_standalone-docker-compose.md)
