## TrustRAG快速上手教程

## 🛠️ 安装

### 方法1：使用`pip`安装

1. 创建conda环境（可选）

```sehll
conda create -n gomate python=3.9
conda activate gomate
```

2. 使用`pip`安装依赖

```sehll
pip install gomate   
```

### 方法2：源码安装

1. 下载源码

```shell
git clone https://github.com/gomate-community/TrustRAG.git
```

2. 安装依赖

```shell
pip install -e . 
```

## 🚀 快速上手

### 1 模块介绍📝

```text
├── applications
├── modules
|      ├── citation:答案与证据引用
|      ├── document：文档解析与切块，支持多种文档类型
|      ├── generator：生成器
|      ├── judger：文档选择
|      ├── prompt：提示语
|      ├── refiner：信息总结
|      ├── reranker：排序模块
|      ├── retrieval：检索模块
|      └── rewriter：改写模块
```


### 2 导入模块

```python
import pickle
import pandas as pd
from tqdm import tqdm

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.txt_parser import TextParser
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.generator.llm import GLM4Chat
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from trustrag.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig
from trustrag.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
```


### 3 文档解析以及切片

```text
def generate_chunks():
    tp = TextParser()# 代表txt格式解析
    tc = TextChunker()
    paragraphs = tp.parse(r'H:/2024-Xfyun-RAG/data/corpus.txt', encoding="utf-8")
    print(len(paragraphs))
    chunks = []
    for content in tqdm(paragraphs):
        chunk = tc.chunk_sentences([content], chunk_size=1024)
        chunks.append(chunk)

    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
```
>corpus.txt每行为一段新闻，可以自行选取paragraph读取的逻辑,语料来自[大模型RAG智能问答挑战赛](https://challenge.xfyun.cn/topic/info?type=RAG-quiz&option=zpsm)

`TextChunker`为文本块切块程序，主要特点使用[InfiniFlow/huqie](https://huggingface.co/InfiniFlow/huqie)作为文本检索的分词器，适合RAG场景。


### 4 构建检索器

**配置检索器：**

下面是一个混合检索器`HybridRetriever`配置参考，其中`HybridRetrieverConfig`需要由`BM25RetrieverConfig`和`DenseRetrieverConfig`配置构成。

```python
# BM25 and Dense Retriever configurations
bm25_config = BM25RetrieverConfig(
    method='lucene',
    index_path='indexs/description_bm25.index',
    k1=1.6,
    b=0.7
)
bm25_config.validate()
print(bm25_config.log_config())
dense_config = DenseRetrieverConfig(
    model_name_or_path=embedding_model_path,
    dim=1024,
    index_path='indexs/dense_cache'
)
config_info = dense_config.log_config()
print(config_info)
# Hybrid Retriever configuration
# 由于分数框架不在同一维度，建议可以合并
hybrid_config = HybridRetrieverConfig(
    bm25_config=bm25_config,
    dense_config=dense_config,
    bm25_weight=0.7,  # bm25检索结果权重
    dense_weight=0.3  # dense检索结果权重
)
hybrid_retriever = HybridRetriever(config=hybrid_config)
```

**构建索引：**

````python
# 构建索引
hybrid_retriever.build_from_texts(corpus)
# 保存索引
hybrid_retriever.save_index()
````

如果构建好索引之后，可以多次使用，直接跳过上面步骤，加载索引
```text
hybrid_retriever.load_index()
```

**检索测试：**

```python
query = "支付宝"
results = hybrid_retriever.retrieve(query, top_k=10)
print(len(results))
# Output results
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']}")
```

### 5 排序模型
```python
reranker_config = BgeRerankerConfig(
    model_name_or_path=reranker_model_path
)
bge_reranker = BgeReranker(reranker_config)
```
### 6 生成器配置
```python
glm4_chat = GLM4Chat(llm_model_path)
```

### 6 检索问答

```python
# ====================检索问答=========================
test = pd.read_csv(test_path)
answers = []
for question in tqdm(test['question'], total=len(test)):
    search_docs = hybrid_retriever.retrieve(question, top_k=10)
    search_docs = bge_reranker.rerank(
        query=question,
        documents=[doc['text'] for idx, doc in enumerate(search_docs)]
    )
    # print(search_docs)
    content = '\n'.join([f'信息[{idx}]：' + doc['text'] for idx, doc in enumerate(search_docs)])
    answer = glm4_chat.chat(prompt=question, content=content)
    answers.append(answer[0])
    print(question)
    print(answer[0])
    print("************************************/n")
test['answer'] = answers

test[['answer']].to_csv(f'{PROJECT_BASE}/output/gomate_baseline.csv', index=False)
```