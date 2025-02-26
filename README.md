# TrustRAG:The RAG Framework within Reliable input,Trusted output
A Configurable and Modular RAG Framework.

\[ English | [中文](README_zh.md) \]


[![Python](https://img.shields.io/badge/Python-3.10.0-3776AB.svg?style=flat)](https://www.python.org)
![workflow status](https://github.com/gomate-community/rageval/actions/workflows/makefile.yml/badge.svg)
[![codecov](https://codecov.io/gh/gomate-community/TrustRAG/graph/badge.svg?token=eG99uSM8mC)](https://codecov.io/gh/gomate-community/TrustRAG)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)


## 🔥Introduction to TrustRAG

TrustRAG is a configurable and modular Retrieval-Augmented Generation (RAG) framework designed to provide **reliable input and trusted output**, ensuring users can obtain high-quality and trustworthy results in retrieval-based question-answering scenarios.

The core design of TrustRAG lies in its **high configurability and modularity**, allowing users to flexibly adjust and optimize each component according to specific needs to meet the requirements of various application scenarios.

## 🔨TrustRAG Framework

![framework.png](resources%2Fframework.png)

## ✨Key Features

**“Reliable input, Trusted output”**

## 🎉 Changelog

- 2025.2.26 add llm_citation
- 2025.2.18 Supports OpenAI DEMO
- Supports vector database engines, such as Milvus and Qdrant.
- Support for multimodal RAG question-answering, API using **GLM-4V-Flash**, code available at [trustrag/applications/rag_multimodal.py](trustrag/applications/rag_multimodal.py)
- TrustRAG packaging and build, supporting both pip and source installation
- Added [MinerU document parsing](https://github.com/gomate-community/TrustRAG/blob/main/docs/mineru.md): A one-stop open-source high-quality data extraction tool, supporting PDF/webpage/multi-format e-book extraction `[20240907]`
- RAPTOR: Recursive tree retriever implementation
- Support for multiple file parsing and modularity, currently supported file types include: `text`, `docx`, `ppt`, `excel`, `html`, `pdf`, `md`, etc.
- Optimized `DenseRetriever`, supporting index building, incremental appending, and index saving, including saving documents, vectors, and indexes
- Added `ReRank` with BGE sorting, Rewriter with `HyDE`
- Added `Judge` with BgeJudge, determining the usefulness of articles `20240711`

## 🚀Quick Start

## 🛠️ Installation

### Method 1: Install via `pip`

1. Create a conda environment (optional)

```shell
conda create -n trustrag python=3.9
conda activate trustrag
```

2. Install dependencies using `pip`

```shell
pip install trustrag   
```

### Method 2: Install from source

1. Download the source code

```shell
git clone https://github.com/gomate-community/TrustRAG.git
```

2. Install dependencies

```shell
pip install -e . 
```

## 🚀 Quick Start

### 1 Module Overview📝

```text
├── applications
├── modules
|      ├── citation: Answer and evidence citation
|      ├── document: Document parsing and chunking, supports multiple document types
|      ├── generator: Generator
|      ├── judger: Document selection
|      ├── prompt: Prompts
|      ├── refiner: Information summarization
|      ├── reranker: Ranking module
|      ├── retrieval: Retrieval module
|      └── rewriter: Rewriting module
```

### 2 Import Modules

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

### 3 Document Parsing and Chunking

```text
def generate_chunks():
    tp = TextParser()  # Represents txt format parsing
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
> Each line in `corpus.txt` is a news paragraph. You can customize the logic for reading paragraphs. The corpus is from [Large Model RAG Intelligent Question-Answering Challenge](https://challenge.xfyun.cn/topic/info?type=RAG-quiz&option=zpsm).

`TextChunker` is the text chunking program, primarily using [InfiniFlow/huqie](https://huggingface.co/InfiniFlow/huqie) as the text retrieval tokenizer, suitable for RAG scenarios.

### 4 Building the Retriever

**Configuring the Retriever:**

Below is a reference configuration for a hybrid retriever `HybridRetriever`, where `HybridRetrieverConfig` is composed of `BM25RetrieverConfig` and `DenseRetrieverConfig`.

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
# Since the score frameworks are not on the same dimension, it is recommended to merge them
hybrid_config = HybridRetrieverConfig(
    bm25_config=bm25_config,
    dense_config=dense_config,
    bm25_weight=0.7,  # BM25 retrieval result weight
    dense_weight=0.3  # Dense retrieval result weight
)
hybrid_retriever = HybridRetriever(config=hybrid_config)
```

**Building the Index:**

````python
# Build the index
hybrid_retriever.build_from_texts(corpus)
# Save the index
hybrid_retriever.save_index()
````

If the index is already built, you can skip the above steps and directly load the index:
```text
hybrid_retriever.load_index()
```

**Retrieval Test:**

```python
query = "Alipay"
results = hybrid_retriever.retrieve(query, top_k=10)
print(len(results))
# Output results
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']}")
```

### 5 Ranking Model
```python
reranker_config = BgeRerankerConfig(
    model_name_or_path=reranker_model_path
)
bge_reranker = BgeReranker(reranker_config)
```
### 6 Generator Configuration
```python
glm4_chat = GLM4Chat(llm_model_path)
```

### 6 Retrieval Question-Answering

```python
# ====================Retrieval Question-Answering=========================
test = pd.read_csv(test_path)
answers = []
for question in tqdm(test['question'], total=len(test)):
    search_docs = hybrid_retriever.retrieve(question, top_k=10)
    search_docs = bge_reranker.rerank(
        query=question,
        documents=[doc['text'] for idx, doc in enumerate(search_docs)]
    )
    # print(search_docs)
    content = '\n'.join([f'Information[{idx}]：' + doc['text'] for idx, doc in enumerate(search_docs)])
    answer = glm4_chat.chat(prompt=question, content=content)
    answers.append(answer[0])
    print(question)
    print(answer[0])
    print("************************************/n")
test['answer'] = answers

test[['answer']].to_csv(f'{PROJECT_BASE}/output/gomate_baseline.csv', index=False)
```

## 🔧Customizing RAG

> Building a custom RAG application

```python
import os

from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.generator.llm import GLMChat
from trustrag.modules.reranker.bge_reranker import BgeReranker
from trustrag.modules.retrieval.dense_retriever import DenseRetriever


class RagApplication():
    def __init__(self, config):
        pass

    def init_vector_store(self):
        pass

    def load_vector_store(self):
        pass

    def add_document(self, file_path):
        pass

    def chat(self, question: str = '', topk: int = 5):
        pass
```

The module can be found at [rag.py](trustrag/applications/rag.py)

### 🌐Experience RAG Effects

You can configure the local model path

```text
# Modify to your own configuration!!!
app_config = ApplicationConfig()
app_config.docs_path = "./docs/"
app_config.llm_model_path = "/data/users/searchgpt/pretrained_models/chatglm3-6b/"

retriever_config = DenseRetrieverConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
    dim=1024,
    index_dir='/data/users/searchgpt/yq/TrustRAG/examples/retrievers/dense_cache'
)
rerank_config = BgeRerankerConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
)

app_config.retriever_config = retriever_config
app_config.rerank_config = rerank_config
application = RagApplication(app_config)
application.init_vector_store()
```

```shell
python app.py
```

Access via browser: [127.0.0.1:7860](127.0.0.1:7860)

![corpus_demo.png](resources%2Fcorpus_demo.png)
![chat_demo.png](resources%2Fchat_demo.png)

App backend logs:
![app_logging3.png](resources%2Fapp_logging3.png)

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gomate-community/TrustRAG&type=Date)](https://star-history.com/#gomate-community/TrustRAG&Date)

## Research and Development Team

This project is completed by the [`GoMate`](https://github.com/gomate-community) team from the Key Laboratory of Network Data Science and Technology, under the guidance of researchers Jiafeng Guo and Yixing Fan.

## Technical Exchange Group

Welcome to provide suggestions and report bad cases. Join the group for timely communication, and PRs are also welcome.</br>

<img src="https://raw.githubusercontent.com/gomate-community/TrustRAG/pipeline/resources/trustrag_group.png" width="180px">

If the group is full or for cooperation and exchange, please contact:

<img src="https://raw.githubusercontent.com/yanqiangmiffy/Chinese-LangChain/master/images/personal.jpg" width="180px">

## 💗Acknowledgments
>This project thanks the following open-source projects for their support and contributions:
- Document parsing: [infiniflow/ragflow](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md)
- PDF file parsing: [opendatalab/MinerU](https://github.com/opendatalab/MinerU)


## 👉 Citation
```text
@article{fan2025trustrag,
  title={TrustRAG: An Information Assistant with Retrieval Augmented Generation},
  author={Fan, Yixing and Yan, Qiang and Wang, Wenshan and Guo, Jiafeng and Zhang, Ruqing and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2502.13719},
  year={2025},
  url={https://arxiv.org/abs/2502.13719}
}
```
