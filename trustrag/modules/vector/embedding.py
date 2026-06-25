import os
from typing import List, Dict
from typing import Optional

import numpy as np
import requests
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModel, AutoTokenizer

from trustrag.modules.vector.base import EmbeddingGenerator


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


class CustomServerEmbedding(EmbeddingGenerator):
    """
    Implementation of EmbeddingGenerator that uses a remote embedding service.
    """

    def __init__(
            self,
            api_url: str = "http://10.208.63.29:6008/v1/embeddings",
            api_key: str = "sk-aaabbbcccdddeeefffggghhhiiijjjkkk",
            model_name: str = "bge-large-en-v1.5",
            timeout: int = 30,
            embedding_size=1024,
    ):
        """
        Initialize the CustomServerEmbedding.

        Args:
            api_url (str): URL of the embedding API
            api_key (str): API key for authentication
            model_name (str): Name of the model to use for embeddings
            timeout (int): Request timeout in seconds
        """
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model_name = model_name
        self.timeout = timeout
        # We don't know the embedding dimension until we make a request
        self.embedding_size = embedding_size

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts by sending a request to the embedding API.

        Args:
            texts (List[str]): List of text strings to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([])

        payload = {
            "input": texts,
            "model": self.model_name
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()  # Raise exception for HTTP errors

            data = response.json()

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]

            # Set embedding size if not yet set
            if self.embedding_size is None and embeddings:
                self.embedding_size = len(embeddings[0])

            return np.array(embeddings)

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to embedding API: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Error parsing embedding API response: {str(e)}")

    def get_token_usage(self, texts: List[str]) -> Dict[str, int]:
        """
        Get token usage statistics for a list of texts.

        Args:
            texts (List[str]): List of text strings to get token usage for

        Returns:
            Dict[str, int]: Dictionary with token usage statistics
        """
        if not texts:
            return {"prompt_tokens": 0, "total_tokens": 0}

        payload = {
            "input": texts,
            "model": self.model_name
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            return data.get("usage", {"prompt_tokens": 0, "total_tokens": 0})

        except Exception:
            return {"prompt_tokens": 0, "total_tokens": 0}


class TwelveLabsEmbedding(EmbeddingGenerator):
    """
    Multimodal embeddings powered by TwelveLabs Marengo.

    Marengo produces embeddings in a single 512-dimensional space that is
    shared across text, image, audio and video. This means a text query and a
    video clip can be compared directly with cosine similarity, which makes it
    a natural fit for video-RAG: index your videos once, then retrieve them
    with plain-text questions.

    This class implements the standard ``generate_embeddings`` text interface
    so it can be used as a drop-in ``EmbeddingGenerator`` for retrieval, and
    additionally exposes ``embed_image``/``embed_audio`` for the other
    modalities. Get a free API key at https://twelvelabs.io.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "marengo3.0",
    ):
        """
        Initialize the TwelveLabs Marengo embedding generator.

        Args:
            api_key (str): TwelveLabs API key. Falls back to the
                ``TWELVELABS_API_KEY`` environment variable.
            model_name (str): Marengo model to use (default ``marengo3.0``).
        """
        from twelvelabs import TwelveLabs

        self.client = TwelveLabs(api_key=api_key or os.getenv("TWELVELABS_API_KEY"))
        self.model_name = model_name
        # Marengo embeddings are 512-dimensional and shared across modalities.
        self.embedding_size = 512

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _embed_text(self, text: str) -> List[float]:
        response = self.client.embed.create(model_name=self.model_name, text=text)
        return response.text_embedding.segments[0].float_

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate Marengo text embeddings for a list of texts.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            np.ndarray: Array of shape (len(texts), 512).
        """
        return np.array([self._embed_text(text) for text in texts])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed_image(self, image_url: str) -> np.ndarray:
        """
        Embed an image into the shared 512-dim Marengo space.

        Args:
            image_url (str): Publicly accessible image URL.

        Returns:
            np.ndarray: Embedding vector with shape (512,).
        """
        response = self.client.embed.create(model_name=self.model_name, image_url=image_url)
        return np.array(response.image_embedding.segments[0].float_)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed_audio(self, audio_url: str) -> np.ndarray:
        """
        Embed audio into the shared 512-dim Marengo space.

        Args:
            audio_url (str): Publicly accessible audio URL.

        Returns:
            np.ndarray: Embedding vector with shape (512,).
        """
        response = self.client.embed.create(model_name=self.model_name, audio_url=audio_url)
        return np.array(response.audio_embedding.segments[0].float_)


class TwelveLabsVideoAnalyzer:
    """
    Video understanding powered by TwelveLabs Pegasus.

    Given a video (by public URL, uploaded asset id, or base64 string), Pegasus
    generates natural-language text describing the video. This is useful in a
    RAG pipeline for turning videos into searchable/grounded text passages
    (summaries, transcripts-with-context, answers to questions about a clip).

    This is intentionally not an ``EmbeddingGenerator`` — it produces text, not
    vectors. Get a free API key at https://twelvelabs.io.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "pegasus1.5",
    ):
        """
        Initialize the TwelveLabs Pegasus video analyzer.

        Args:
            api_key (str): TwelveLabs API key. Falls back to the
                ``TWELVELABS_API_KEY`` environment variable.
            model_name (str): Pegasus model to use (default ``pegasus1.5``).
        """
        from twelvelabs import TwelveLabs

        self.client = TwelveLabs(api_key=api_key or os.getenv("TWELVELABS_API_KEY"))
        self.model_name = model_name

    def analyze(
            self,
            prompt: str,
            video_url: Optional[str] = None,
            video_id: Optional[str] = None,
            asset_id: Optional[str] = None,
            max_tokens: int = 2048,
    ) -> str:
        """
        Generate text from a video for a given prompt.

        Exactly one of ``video_url``, ``video_id`` or ``asset_id`` must be set.

        Args:
            prompt (str): Instruction, e.g. "Summarize this video".
            video_url (str): Publicly accessible video URL.
            video_id (str): Id of a video already indexed in TwelveLabs.
            asset_id (str): Id of an uploaded TwelveLabs asset.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        from twelvelabs.types.video_context import VideoContext_AssetId, VideoContext_Url

        kwargs: Dict = {"model_name": self.model_name, "prompt": prompt, "max_tokens": max_tokens}
        if video_id is not None:
            kwargs["video_id"] = video_id
        elif video_url is not None:
            kwargs["video"] = VideoContext_Url(url=video_url)
        elif asset_id is not None:
            kwargs["video"] = VideoContext_AssetId(asset_id=asset_id)
        else:
            raise ValueError("One of video_url, video_id or asset_id must be provided")

        response = self.client.analyze(**kwargs)
        return response.data


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
        elif embedding_type == 'twelvelabs':
            return TwelveLabsEmbedding(
                api_key=kwargs.get('api_key'),
                model_name=kwargs.get('model_name', 'marengo3.0')
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
            'twelvelabs',
            'flag_model'
        ]
