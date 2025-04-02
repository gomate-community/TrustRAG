from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
from trustrag.modules.vector.embedding import EmbeddingGenerator
from typing import Union
class MilvusEngine:
    def __init__(
        self,
        collection_name: str,
        embedding_generator: EmbeddingGenerator,
        milvus_client_params: Dict[str, Any] = {"uri": "./milvus_demo.db"},
        vector_size: int = 1536,
        metric_type: str = "IP",  # Inner product distance
        consistency_level: str = "Strong",  # Strong consistency level
    ):
        """
        Initialize the Milvus vector store.

        :param collection_name: Name of the Milvus collection.
        :param embedding_generator: An instance of EmbeddingGenerator to generate embeddings.
        :param milvus_client_params: Dictionary of parameters to pass to MilvusClient.
        :param vector_size: Size of the vectors.
        :param metric_type: Distance metric for vector comparison (default is inner product).
        :param consistency_level: Consistency level for the collection (default is strong).
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.metric_type = metric_type
        self.consistency_level = consistency_level
        self.embedding_generator = embedding_generator

        # Initialize MilvusClient with provided parameters
        self.client = MilvusClient(**milvus_client_params)

        # Create collection if it doesn't exist
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_size,
                metric_type=self.metric_type,
                consistency_level=self.consistency_level,
            )

    def upload_vectors(
        self, vectors: Union[np.ndarray, List[List[float]]],
        payload: List[Dict[str, Any]],
        batch_size: int = 256
    ):
        """
        Upload vectors and payload to the Milvus collection.

        :param vectors: A numpy array or list of vectors to upload.
        :param payload: A list of dictionaries containing the payload for each vector.
        :param batch_size: Number of vectors to upload in a single batch.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if len(vectors) != len(payload):
            raise ValueError("Vectors and payload must have the same length.")

        data = []
        for i, (vector, payload_item) in enumerate(zip(vectors, payload)):
            data.append({"id": i, "vector": vector.tolist(), **payload_item})

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(
        self, text: str,
        query_filter: str = None,
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
        vector = self.embedding_generator.generate_embeddings([text])

        # Search for closest vectors in the collection
        search_result = self.client.search(
            collection_name=self.collection_name,
            data=[vector[0]],  # Use the first (and only) embedding
            limit=limit,
            search_params={"metric_type": self.metric_type, "params": {}},
            output_fields=["*"],  # Return all fields
            filter=query_filter,
        )

        # Extract payloads from search results
        payloads = [hit["entity"] for hit in search_result[0]]
        return payloads

    def build_filter(self, conditions: List[Dict[str, Any]]) -> str:
        """
        Build a Milvus filter from a list of conditions.

        :param conditions: A list of conditions, where each condition is a dictionary with:
                          - key: The field name to filter on.
                          - value: The value to match (can be a string, number, or other supported types).
        :return: A Milvus filter dictionary.
        """
        filter_conditions = []
        for condition in conditions:
            key = condition.get("key")
            value = condition.get("value")
            if key and value is not None:
                filter_conditions.append(f"{key} == '{value}'")

        return " and ".join(filter_conditions) if filter_conditions else None

