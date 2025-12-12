# 步骤一：安装依赖库
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType

dashscope_api_key = "<YOUR_DASHSCOPE_API_KEY>"
milvus_url = "<YOUR_MMILVUS_URL>"
user_name = "root"
password = "<YOUR_PASSWORD>"
collection_name = "milvus_overview"
dense_dim = 1536

# 步骤二：数据准备
loader = WebBaseLoader([
    'https://raw.githubusercontent.com/milvus-io/milvus-docs/refs/heads/v2.5.x/site/en/about/overview.md'
])

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)

# 使用LangChain将输入文档安照chunk_size切分
all_splits = text_splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2", dashscope_api_key=dashscope_api_key
)

text_contents = [doc.page_content for doc in all_splits]

vectors = embeddings.embed_documents(text_contents)


client = MilvusClient(
    uri=f"http://{milvus_url}:19530",
    token=f"{user_name}:{password}",
)

schema = MilvusClient.create_schema(
    enable_dynamic_field=True,
)

analyzer_params = {
    "type": "english"
}

# Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=analyzer_params, enable_match=True)
schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)

bm25_function = Function(
   name="bm25",
   function_type=FunctionType.BM25,
   input_field_names=["text"],
   output_field_names="sparse_bm25",
)
schema.add_function(bm25_function)

index_params = client.prepare_index_params()

# Add indexes
index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128},
)

index_params.add_index(
    field_name="sparse_bm25",
    index_name="sparse_bm25_index",
    index_type="SPARSE_WAND",
    metric_type="BM25"
)

# Create collection
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

data = [
    {"dense": vectors[idx], "text": doc}
    for idx, doc in enumerate(text_contents)
]

# Insert data
res = client.insert(
    collection_name=collection_name,
    data=data
)

print(f"生成 {len(vectors)} 个向量，维度：{len(vectors[0])}")

# 同样，在处理中文文档时，Milvus 2.5版本也支持指定相应的中文分析器。
# # 定义分词器参数
# analyzer_params = {
#     "type": "chinese"  # 指定分词器类型为中文
# }
#
# # 添加文本字段到 Schema，并启用分词器
# schema.add_field(
#     field_name="text",                      # 字段名称
#     datatype=DataType.VARCHAR,              # 数据类型：字符串（VARCHAR）
#     max_length=65535,                       # 最大长度：65535 字符
#     enable_analyzer=True,                   # 启用分词器
#     analyzer_params=analyzer_params         # 分词器参数
# )

# 步骤三：全文检索
from pymilvus import MilvusClient

# 创建Milvus Client。
client = MilvusClient(
    uri="http://c-xxxx.milvus.aliyuncs.com:19530",  # Milvus实例的公网地址。
    token="<yourUsername>:<yourPassword>",  # 登录Milvus实例的用户名和密码。
    db_name="default"  # 待连接的数据库名称，本文示例为默认的default。
)

search_params = {
    'params': {'drop_ratio_search': 0.2},
}

full_text_search_res = client.search(
    collection_name='milvus_overview',
    data=['what makes milvus so fast?'],
    anns_field='sparse_bm25',
    limit=3,
    search_params=search_params,
    output_fields=["text"],
)

for hits in full_text_search_res:
    for hit in hits:
        print(hit)
        print("\n")

# 步骤四：关键词匹配
# filter = "TEXT_MATCH(text, 'query') and TEXT_MATCH(text, 'node')"
#
# text_match_res = client.search(
#     collection_name="milvus_overview",
#     anns_field="dense",
#     data=query_embeddings,
#     filter=filter,
#     search_params={"params": {"nprobe": 10}},
#     limit=2,
#     output_fields=["text"]
# )

# 步骤五：混合检索与RAG
from pymilvus import MilvusClient
from pymilvus import AnnSearchRequest, RRFRanker
from langchain_community.embeddings import DashScopeEmbeddings
from dashscope import Generation

# 创建Milvus Client。
client = MilvusClient(
    uri="http://c-xxxx.milvus.aliyuncs.com:19530",  # Milvus实例的公网地址。
    token="<yourUsername>:<yourPassword>",  # 登录Milvus实例的用户名和密码。
    db_name="default"  # 待连接的数据库名称，本文示例为默认的default。
)

collection_name = "milvus_overview"

# 替换为您的 DashScope API-KEY
dashscope_api_key = "<YOUR_DASHSCOPE_API_KEY>"

# 初始化 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 使用text-embedding-v2模型。
    dashscope_api_key=dashscope_api_key
)

# Define the query
query = "Why does Milvus run so scalable?"

# Embed the query and generate the corresponding vector representation
query_embeddings = embeddings.embed_documents([query])

# Set the top K result count
top_k = 5  # Get the top 5 docs related to the query

# Define the parameters for the dense vector search
search_params_dense = {
    "metric_type": "IP",
    "params": {"nprobe": 2}
}

# Create a dense vector search request
request_dense = AnnSearchRequest([query_embeddings[0]], "dense", search_params_dense, limit=top_k)

# Define the parameters for the BM25 text search
search_params_bm25 = {
    "metric_type": "BM25"
}

# Create a BM25 text search request
request_bm25 = AnnSearchRequest([query], "sparse_bm25", search_params_bm25, limit=top_k)

# Combine the two requests
reqs = [request_dense, request_bm25]

# Initialize the RRF ranking algorithm
ranker = RRFRanker(100)

# Perform the hybrid search
hybrid_search_res = client.hybrid_search(
    collection_name=collection_name,
    reqs=reqs,
    ranker=ranker,
    limit=top_k,
    output_fields=["text"]
)

# Extract the context from hybrid search results
context = []
print("Top K Results:")
for hits in hybrid_search_res:  # Use the correct variable here
    for hit in hits:
        context.append(hit['entity']['text'])  # Extract text content to the context list
        print(hit['entity']['text'])  # Output each retrieved document


# Define a function to get an answer based on the query and context
def getAnswer(query, context):
    prompt = f'''Please answer my question based on the content within:
    ```
    {context}
    ```
    My question is: {query}.
    '''
    # Call the generation module to get an answer
    rsp = Generation.call(model='qwen-turbo', prompt=prompt)
    return rsp.output.text

# Get the answer
answer = getAnswer(query, context)

print(answer)


# Expected output excerpt
"""
Milvus is highly scalable due to its cloud-native and highly decoupled system architecture. This architecture allows the system to continuously expand as data grows. Additionally, Milvus supports three deployment modes that cover a wide...
"""