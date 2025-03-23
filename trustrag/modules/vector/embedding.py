import os
from typing import List, Optional

import numpy as np
import torch
from FlagEmbedding import FlagAutoModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModel, AutoTokenizer

from trustrag.modules.vector.base import EmbeddingGenerator


class OpenAIEmbedding(EmbeddingGenerator):
    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            embedding_model_name: str = "text-embedding-3-large"
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = embedding_model_name

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        return np.array([data.embedding for data in response.data])


class SentenceTransformerEmbedding(EmbeddingGenerator):
    def __init__(
            self,
            model_name_or_path: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
            device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self.embedding_size = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class HuggingFaceEmbedding(EmbeddingGenerator):
    def __init__(
            self,
            model_name: str,
            device: str = None,
            trust_remote_code: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_input)
            embeddings = outputs[0][:, 0]  # Use CLS token embeddings

        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()


class ZhipuEmbedding(EmbeddingGenerator):
    def __init__(self, api_key: str = None, model: str = "embedding-2"):
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key or os.getenv("ZHIPUAI_API_KEY"))
        self.model = model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([data.embedding for data in response.data])


class DashscopeEmbedding(EmbeddingGenerator):
    def __init__(self, api_key: str = None, model: str = "text-embedding-v1"):
        import dashscope
        dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.client = dashscope.TextEmbedding
        self.model = model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = self.client.call(
                model=self.model,
                input=text
            )
            embeddings.append(response.output['embeddings'][0]['embedding'])
        return np.array(embeddings)


class FlagModelEmbedding(EmbeddingGenerator):
    def __init__(
            self,
            model_name: str = "BAAI/bge-base-en-v1.5",
            query_instruction: Optional[str] = "Represent this sentence for searching relevant passages:",
            use_fp16: bool = True,
            device: str = None
    ):
        """
        Initialize FlagModel embedding generator.

        Args:
            model_name (str): Name or path of the model
            query_instruction (str, optional): Instruction prefix for queries
            use_fp16 (bool): Whether to use FP16 for inference
            device (str, optional): Device to run the model on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlagAutoModel.from_finetuned(
            model_name,
            query_instruction_for_retrieval=query_instruction,
            use_fp16=use_fp16,
            devices=self.device
        )
        # if self.device == "cuda":
        #     self.model.to(device)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = self.model.encode(texts)
        return np.array(embeddings)

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings using inner product.

        Args:
            embeddings1 (np.ndarray): First set of embeddings
            embeddings2 (np.ndarray): Second set of embeddings

        Returns:
            np.ndarray: Similarity matrix
        """
        return embeddings1 @ embeddings2.T


class EmbeddingFactory:
    """
    工厂类，用于创建和管理不同类型的嵌入生成器。
    支持OpenAI、SentenceTransformer、HuggingFace、Zhipu、Dashscope和FlagModel等多种嵌入模型。
    """

    @staticmethod
    def create_embedding_generator(
            embedding_type: str,
            **kwargs
    ) -> EmbeddingGenerator:
        """
        根据指定的嵌入类型创建相应的嵌入生成器实例。

        Args:
            embedding_type (str): 嵌入生成器类型，可选值包括：
                                 'openai', 'sentence_transformer', 'huggingface',
                                 'zhipu', 'dashscope', 'flag_model'
            **kwargs: 传递给具体嵌入生成器构造函数的参数

        Returns:
            EmbeddingGenerator: 创建的嵌入生成器实例

        Raises:
            ValueError: 当指定的嵌入类型不受支持时
        """
        embedding_type = embedding_type.lower()

        if embedding_type == 'openai':
            return OpenAIEmbedding(
                api_key=kwargs.get('api_key'),
                base_url=kwargs.get('base_url'),
                embedding_model_name=kwargs.get('model_name', 'text-embedding-3-large')
            )
        elif embedding_type == 'sentence_transformer':
            return SentenceTransformerEmbedding(
                model_name_or_path=kwargs.get('model_name', 'sentence-transformers/multi-qa-mpnet-base-cos-v1'),
                device=kwargs.get('device')
            )
        elif embedding_type == 'huggingface':
            if 'model_name' not in kwargs:
                raise ValueError("必须为HuggingFace嵌入提供'model_name'参数")
            return HuggingFaceEmbedding(
                model_name=kwargs['model_name'],
                device=kwargs.get('device'),
                trust_remote_code=kwargs.get('trust_remote_code', True)
            )
        elif embedding_type == 'zhipu':
            return ZhipuEmbedding(
                api_key=kwargs.get('api_key'),
                model=kwargs.get('model_name', 'embedding-2')
            )
        elif embedding_type == 'dashscope':
            return DashscopeEmbedding(
                api_key=kwargs.get('api_key'),
                model=kwargs.get('model_name', 'text-embedding-v1')
            )
        elif embedding_type == 'flag_model':
            return FlagModelEmbedding(
                model_name=kwargs.get('model_name', 'BAAI/bge-base-en-v1.5'),
                query_instruction=kwargs.get('query_instruction',
                                             'Represent this sentence for searching relevant passages:'),
                use_fp16=kwargs.get('use_fp16', True),
                device=kwargs.get('device')
            )
        else:
            raise ValueError(f"不支持的嵌入类型: {embedding_type}")

    @staticmethod
    def get_available_embedding_types() -> List[str]:
        """
        获取所有可用的嵌入类型。

        Returns:
            List[str]: 可用嵌入类型列表
        """
        return [
            'openai',
            'sentence_transformer',
            'huggingface',
            'zhipu',
            'dashscope',
            'flag_model'
        ]
