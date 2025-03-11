import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagAutoModel


class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]

    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return 0 if not magnitude else dot_product / magnitude


class OpenAIEmbedding(EmbeddingGenerator):
    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
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