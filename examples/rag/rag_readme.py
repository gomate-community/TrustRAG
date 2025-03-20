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