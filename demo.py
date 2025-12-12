#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
最简化的RAG问答系统
直接执行，无函数封装
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(".")
from tqdm import tqdm

from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.vector.embedding import SentenceTransformerEmbedding
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig, DenseRetriever
from trustrag.modules.generator.llm import Qwen3Chat

# 配置参数
DOCS_PATH = r"data/docs"
LLM_MODEL_PATH = r"G:\pretrained_models\llm\Qwen3-4B"
EMBEDDING_MODEL_PATH = r"G:\pretrained_models\mteb\bge-large-zh-v1.5"
RERANKER_MODEL_PATH = r"G:\pretrained_models\mteb\bge-reranker-large"
INDEX_PATH = r"examples/retrievers/dense_cache"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 256
TOP_K = 5

print("🚀 启动RAG问答系统")
print("="*50)

# Step 1: 初始化组件
print("Step 1: 正在初始化组件...")

# 初始化文档解析器
parser = CommonParser()
print("  ✓ 文档解析器初始化完成")

# 初始化文本分块器
tc = TextChunker()
print("  ✓ 文本分块器初始化完成")

# 初始化嵌入模型
embedding_generator = SentenceTransformerEmbedding(EMBEDDING_MODEL_PATH)
print("  ✓ 嵌入模型初始化完成")

# 初始化检索器
retriever_config = DenseRetrieverConfig(
    model_name_or_path=EMBEDDING_MODEL_PATH,
    dim=EMBEDDING_DIM,
    index_path=INDEX_PATH
)
retriever = DenseRetriever(retriever_config, embedding_generator)
print("  ✓ 检索器初始化完成")

# 初始化重排序器
rerank_config = BgeRerankerConfig(
    model_name_or_path=RERANKER_MODEL_PATH
)
reranker = BgeReranker(rerank_config)
print("  ✓ 重排序器初始化完成")

# 初始化大语言模型
llm = Qwen3Chat(LLM_MODEL_PATH)
print("  ✓ 大语言模型初始化完成")

print("Step 1: 所有组件初始化完成!\n")

# Step 2: 处理向量索引
print("Step 2: 处理向量索引...")

# 检查是否存在现有索引
if os.path.exists(INDEX_PATH):
    print("  发现现有索引，正在加载...")
    retriever.load_index(INDEX_PATH)
    print("  ✓ 索引加载完成\n")
else:
    print("  未发现现有索引，开始构建新索引...")
    
    # Step 3: 构建向量存储
    print("Step 3: 构建向量存储...")
    
    # 检查文档目录
    if not os.path.exists(DOCS_PATH):
        print(f"  ❌ 文档目录 {DOCS_PATH} 不存在")
        print(f"  请创建目录并添加文档文件")
        exit(1)
    
    # 获取所有文档文件
    doc_files = [f for f in os.listdir(DOCS_PATH) if os.path.isfile(os.path.join(DOCS_PATH, f))]
    if not doc_files:
        print("  ❌ 文档目录为空，请添加文档文件")
        exit(1)
    
    print(f"  发现 {len(doc_files)} 个文档文件")
    
    # 解析所有文档
    all_paragraphs = []
    for filename in doc_files:
        file_path = os.path.join(DOCS_PATH, filename)
        try:
            paragraphs = parser.parse(file_path)
            all_paragraphs.append(paragraphs)
            print(f"  ✓ 已解析: {filename}")
        except Exception as e:
            print(f"  ❌ 解析失败 {filename}: {e}")
    
    if not all_paragraphs:
        print("  ❌ 没有成功解析的文档")
        exit(1)
    
    # 文档分块
    print("  正在进行文档分块...")
    all_chunks = []
    for paragraphs in tqdm(all_paragraphs, desc="  分块处理"):
        if isinstance(paragraphs, list) and paragraphs:
            if isinstance(paragraphs[0], dict):
                text_list = [' '.join(str(value) for value in item.values()) for item in paragraphs]
            else:
                text_list = [str(item) for item in paragraphs]
        else:
            text_list = [str(paragraphs)] if paragraphs else []
        
        chunks = tc.get_chunks(text_list, CHUNK_SIZE)
        all_chunks.extend(chunks)
    
    print(f"  ✓ 生成了 {len(all_chunks)} 个文档块")
    
    # 构建向量索引
    print("  正在构建向量索引...")
    retriever.build_from_texts(all_chunks)
    
    # 保存索引
    index_dir = os.path.dirname(INDEX_PATH)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    retriever.save_index(INDEX_PATH)
    print(f"  ✓ 索引已保存到: {INDEX_PATH}")
    print("Step 3: 向量存储构建完成!\n")

# Step 4: 开始问答循环
print("Step 4: 启动问答系统")
print("="*50)
print("RAG问答系统已启动!")
print("输入 'quit' 或 'exit' 退出程序")
print("输入 'rebuild' 重新构建索引")
print("="*50)

while True:
    try:
        # 获取用户输入
        question = input("\n请输入您的问题: ").strip()
        
        # 检查退出命令
        if question.lower() in ['quit', 'exit', '退出']:
            print("再见!")
            break
        
        # 检查重建索引命令
        if question.lower() in ['rebuild', '重建']:
            print("\n正在重新构建索引...")
            
            # 重新构建索引的代码
            doc_files = [f for f in os.listdir(DOCS_PATH) if os.path.isfile(os.path.join(DOCS_PATH, f))]
            all_paragraphs = []
            for filename in doc_files:
                file_path = os.path.join(DOCS_PATH, filename)
                try:
                    paragraphs = parser.parse(file_path)
                    all_paragraphs.append(paragraphs)
                except:
                    pass
            
            all_chunks = []
            for paragraphs in tqdm(all_paragraphs, desc="重新分块"):
                if isinstance(paragraphs, list) and paragraphs:
                    if isinstance(paragraphs[0], dict):
                        text_list = [' '.join(str(value) for value in item.values()) for item in paragraphs]
                    else:
                        text_list = [str(item) for item in paragraphs]
                else:
                    text_list = [str(paragraphs)] if paragraphs else []
                
                chunks = tc.get_chunks(text_list, CHUNK_SIZE)
                all_chunks.extend(chunks)
            
            retriever.build_from_texts(all_chunks)
            retriever.save_index(INDEX_PATH)
            print("索引重建完成!")
            continue
        
        # 检查空输入
        if not question:
            print("请输入有效的问题")
            continue
        
        print("\n正在思考中...")
        
        # RAG问答处理
        print(f"正在处理问题: {question}")
        
        # 检索相关文档
        print("  正在检索相关文档...")
        contents = retriever.retrieve(query=question, top_k=TOP_K)
        print(f"  ✓ 检索到 {len(contents)} 个相关文档块")
        
        # 重排序
        print("  正在重排序文档...")
        contents = reranker.rerank(query=question, documents=[content['text'] for content in contents])
        print("  ✓ 文档重排序完成")
        
        # 构建上下文
        print("  正在构建上下文...")
        context = '\n'.join([content['text'] for content in contents])
        print("  ✓ 上下文构建完成")
        
        # 生成回答
        print("  正在生成回答...")
        result, history = llm.chat(question, [], context)
        print("  ✓ 回答生成完成")
        
        # 输出结果
        print("\n" + "="*50)
        print("回答:")
        print(result)
        print("\n" + "-"*30)
        print(f"参考了 {len(contents)} 个相关文档片段")
        
        # 可选显示参考文档
        show_sources = input("\n是否显示参考文档片段? (y/n): ").strip().lower()
        if show_sources in ['y', 'yes', '是']:
            print("\n参考文档片段:")
            for idx, source in enumerate(contents[:3], 1):
                score = source.get('score', 0)
                text = source['text']
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n[片段 {idx}] (相关度: {score:.3f})")
                print(preview)
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        break
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请重试或输入 'quit' 退出")

print("\n程序结束")