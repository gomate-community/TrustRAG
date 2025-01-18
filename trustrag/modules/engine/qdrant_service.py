import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from openai import OpenAI

class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        pass

class SentenceTransformerEmbedding(EmbeddingGenerator):
    def __init__(self, model_name_or_path: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name_or_path, device=device)

    def generate_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class OpenAIEmbedding(EmbeddingGenerator):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_embedding(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        return resp.data[0].embedding

class QdrantEngine:
    def __init__(
        self,
        collection_name: str,
        embedding_generator: EmbeddingGenerator,
        qdrant_client_params: Dict[str, Any] = {"host": "localhost", "port": 6333},
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
    ):
        """
        Initialize the Qdrant vector store.

        :param collection_name: Name of the Qdrant collection.
        :param embedding_generator: An instance of EmbeddingGenerator to generate embeddings.
        :param qdrant_client_params: Dictionary of parameters to pass to QdrantClient.
        :param vector_size: Size of the vectors.
        :param distance: Distance metric for vector comparison (default is cosine similarity).
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.embedding_generator = embedding_generator

        # Initialize QdrantClient with provided parameters
        self.client = QdrantClient(**qdrant_client_params)

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
            )

    def upload_vectors(
            self, vectors: Union[np.ndarray, List[List[float]]],
            payload: List[Dict[str, Any]],
            batch_size: int = 256
    ):
        """
        Upload vectors and payload to the Qdrant collection.

        :param vectors: A numpy array or list of vectors to upload.
        :param payload: A list of dictionaries containing the payload for each vector.
        :param batch_size: Number of vectors to upload in a single batch.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if len(vectors) != len(payload):
            raise ValueError("Vectors and payload must have the same length.")
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payload,
            ids=None,
            batch_size=batch_size,
        )

    def search(
            self, text: str,
            query_filter: Optional[Filter] = None,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for the closest vectors in the collection based on the input text.

        :param text: The text query to search for.
        :param query_filter: Optional filter to apply to the search.
        :param limit: Number of closest results to return.
        :return: List of payloads from the closest vectors.
        """
        # Generate embedding using the provided embedding generator
        vector = self.embedding_generator.generate_embedding(text)

        # Search for closest vectors in the collection
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        # Extract payloads from search results
        payloads = [hit.payload for hit in search_result]
        return payloads

    def build_filter(self, conditions: List[Dict[str, Any]]) -> Filter:
        """
        Build a Qdrant filter from a list of conditions.

        :param conditions: A list of conditions, where each condition is a dictionary with:
                          - key: The field name to filter on.
                          - match: The value to match (can be a string, number, or other supported types).
        :return: A Qdrant Filter object.
        """
        filter_conditions = []
        for condition in conditions:
            key = condition.get("key")
            match_value = condition.get("match")
            if key and match_value is not None:
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=match_value),
                    )
                )

        return Filter(must=filter_conditions)

if __name__ == "__main__":
    # Initialize embedding generators
    local_embedding_generator = SentenceTransformerEmbedding(model_name_or_path="all-MiniLM-L6-v2", device="cpu")
    openai_embedding_generator = OpenAIEmbedding(api_key="your_key", base_url="https://ark.cn-beijing.volces.com/api/v3", model="your_model_id")

    # Initialize QdrantEngine with local embedding generator
    vector_store = QdrantEngine(
        collection_name="startups",
        embedding_generator=local_embedding_generator,
        qdrant_client_params={"host": "localhost", "port": 6333},
    )

    # Example vectors and payload
    vectors = np.random.rand(100, 384).tolist()
    payload = [{"name": f"Startup {i}", "city": "Berlin", "category": "AI"} for i in range(100)]

    # Upload vectors and payload
    vector_store.upload_vectors(vectors=vectors, payload=payload)

    # Build a filter for city and category
    conditions = [
        {"key": "city", "match": "Berlin"},
        {"key": "category", "match": "AI"},
    ]
    custom_filter = vector_store.build_filter(conditions)

    # Search for startups related to "AI" in Berlin
    results = vector_store.search(text="AI", query_filter=custom_filter, limit=5)
    for result in results:
        print(result)