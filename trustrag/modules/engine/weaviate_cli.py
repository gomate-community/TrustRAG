from typing import List, Dict, Any, Optional, Union
import numpy as np
import weaviate
from weaviate import WeaviateClient
from weaviate.collections import Collection
import weaviate.classes.config as wc
from weaviate.classes.config import Property, DataType
from trustrag.modules.vector.embedding import EmbeddingGenerator
from  weaviate.classes.query import MetadataQuery

class WeaviateEngine:
    def __init__(
            self,
            collection_name: str,
            embedding_generator: EmbeddingGenerator,
            client_params: Dict[str, Any] = {
                "http_host": "localhost",
                "http_port": 8080,
                "http_secure": False,
                "grpc_host": "localhost",
                "grpc_port": 50051,
                "grpc_secure": False,
            },
    ):
        """
        Initialize the Weaviate vector store.

        :param collection_name: Name of the Weaviate collection
        :param embedding_generator: An instance of EmbeddingGenerator to generate embeddings
        :param weaviate_client_params: Dictionary of parameters to pass to Weaviate client
        """
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator

        # Initialize Weaviate client with provided parameters
        self.client = weaviate.connect_to_custom(
            skip_init_checks=False,
            **client_params
        )

        # Create collection if it doesn't exist
        if not self._collection_exists():
            self._create_collection()

    def _collection_exists(self) -> bool:
        """Check if collection exists in Weaviate."""
        try:
            collections = self.client.collections.list_all()
            collection_names = [c.lower() for c in collections]
            return self.collection_name in collection_names
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False

    def _create_collection(self):
        """Create a new collection in Weaviate."""
        try:
            self.client.collections.create(
                name=self.collection_name,
                # Define properties of metadata
                properties=[
                    wc.Property(
                        name="text",
                        data_type=wc.DataType.TEXT
                    ),
                    wc.Property(
                        name="title",
                        data_type=wc.DataType.TEXT,
                        skip_vectorization=True
                    ),
                ]
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def upload_vectors(
            self,
            vectors: Union[np.ndarray, List[List[float]]],
            payload: List[Dict[str, Any]],
            batch_size: int = 100
    ):
        """
        Upload vectors and payload to the Weaviate collection.

        :param vectors: A numpy array or list of vectors to upload
        :param payload: A list of dictionaries containing the payload for each vector
        :param batch_size: Number of vectors to upload in a single batch
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if len(vectors) != len(payload):
            raise ValueError("Vectors and payload must have the same length.")

        collection = self.client.collections.get(self.collection_name)

        # Process in batches
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payload = payload[i:i + batch_size]

            try:
                with collection.batch.dynamic() as batch:
                    for idx, (properties, vector) in enumerate(zip(batch_payload, batch_vectors)):
                        # Separate text content and other metadata
                        text_content = properties.get('description',
                                                      '')  # Assuming 'description' is the main text field
                        metadata = {k: v for k, v in properties.items() if k != 'description'}

                        # Prepare the properties dictionary
                        properties_dict = {
                            "text": text_content,
                            "title": metadata.get('title', f'Object {idx}')  # Using title from metadata or default
                        }

                        # Add the object with properties and vector
                        batch.add_object(
                            properties=properties_dict,
                            vector=vector
                        )
            except Exception as e:
                print(f"Error uploading batch: {e}")
                raise

    def search(
            self,
            text: str,
            query_filter: Optional[Dict[str, Any]] = None,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for the closest vectors in the collection based on the input text.

        :param text: The text query to search for
        :param query_filter: Optional filter to apply to the search
        :param limit: Number of closest results to return
        :return: List of payloads from the closest vectors
        """
        # Generate embedding for the query text
        vector = self.embedding_generator.generate_embedding(text)
        print(vector.shape)
        collection = self.client.collections.get(self.collection_name)

        # Prepare query arguments
        query_args = {
            "near_vector": vector,
            "limit": limit,
            "return_metadata": MetadataQuery(distance=True)
        }

        # Add filter if provided
        if query_filter:
            query_args["filter"] = query_filter

        results = collection.query.near_vector(**query_args)

            # Convert results to the same format as QdrantEngine
        payloads = []
        for obj in results.objects:
            payload = obj.properties.get('metadata', {})
            payload['text'] = obj.properties.get('text', '')
            payload['_distance'] = obj.metadata.distance
            payloads.append(payload)

        return payloads


    def build_filter(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a Weaviate filter from a list of conditions.

        :param conditions: A list of conditions, where each condition is a dictionary with:
                         - key: The field name to filter on
                         - match: The value to match
        :return: A Weaviate filter object
        """
        filter_dict = {
            "operator": "And",
            "operands": []
        }

        for condition in conditions:
            key = condition.get("key")
            match_value = condition.get("match")
            if key and match_value is not None:
                filter_dict["operands"].append({
                    "path": [f"metadata.{key}"],
                    "operator": "Equal",
                    "valueString": str(match_value)
                })

        return filter_dict if filter_dict["operands"] else None


