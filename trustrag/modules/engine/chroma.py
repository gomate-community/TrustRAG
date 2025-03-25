from typing import List, Dict, Any, Union
import numpy as np
import chromadb
from chromadb.config import Settings
from trustrag.modules.vector.embedding import EmbeddingGenerator


class ChromaEngine:
    def __init__(
            self,
            collection_name: str,
            embedding_generator: EmbeddingGenerator,
            chroma_client_params: Dict[str, Any] = {"path": "./chroma_db"},
            consistency_level: str = "Strong",  # Kept for API compatibility
    ):
        """
        Initialize the Chroma vector store.

        :param collection_name: Name of the Chroma collection.
        :param embedding_generator: An instance of EmbeddingGenerator to generate embeddings.
        :param chroma_client_params: Dictionary of parameters to pass to ChromaClient.
        :param consistency_level: Kept for API compatibility with Milvus.
        """
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(**chroma_client_params)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity by default
        )

    def upload_vectors(
            self,
            vectors: Union[np.ndarray, List[List[float]]],
            payload: List[Dict[str, Any]],
            batch_size: int = 256
    ):
        """
        Upload vectors and payload to the Chroma collection.

        :param vectors: A numpy array or list of vectors to upload.
        :param payload: A list of dictionaries containing the payload for each vector.
        :param batch_size: Number of vectors to upload in a single batch.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if len(vectors) != len(payload):
            raise ValueError("Vectors and payload must have the same length.")

        # Prepare data for Chroma format
        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for i, (vector, payload_item) in enumerate(zip(vectors, payload)):
            # Convert payload to string for documents
            documents.append(payload_item.get('description', ''))
            # Remove description from metadata if it exists
            metadata = payload_item.copy()
            metadata.pop('description', None)
            metadatas.append(metadata)
            ids.append(str(i))
            embeddings.append(vector.tolist())

        # Upload in batches
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end]
            )

    def search(
            self,
            text: str,
            query_filter: Dict[str, Any] = None,
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

        # Search in the collection
        results = self.collection.query(
            query_embeddings=vector[0].tolist(),
            n_results=limit,
            where=query_filter  # Chroma uses dict-based filtering
        )

        # Format results to match Milvus-style output
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'description': results['documents'][0][i],
                **results['metadatas'][0][i]
            }
            formatted_results.append(result)

        return formatted_results

    def build_filter(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a Chroma filter from a list of conditions.

        :param conditions: A list of conditions, where each condition is a dictionary with:
                         - key: The field name to filter on
                         - value: The value to match
                         - operator: The operation to perform
        :return: A Chroma filter dictionary
        """
        filter_dict = {}

        for condition in conditions:
            key = condition.get("key")
            value = condition.get("value")
            operator = condition.get("operator", "==")

            if key and value is not None:
                if operator == "like":
                    # Chroma doesn't support LIKE directly, we use exact match
                    filter_dict[key] = value
                elif operator == ">":
                    filter_dict[key] = {"$gt": value}
                elif operator == "<":
                    filter_dict[key] = {"$lt": value}
                elif operator == ">=":
                    filter_dict[key] = {"$gte": value}
                elif operator == "<=":
                    filter_dict[key] = {"$lte": value}
                else:  # Default to exact match
                    filter_dict[key] = value

        return filter_dict if filter_dict else None