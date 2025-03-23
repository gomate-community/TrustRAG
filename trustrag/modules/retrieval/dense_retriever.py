#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 23:07
"""
import gc
import os
from typing import List, Dict, Union
import numpy as np
import faiss
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from trustrag.modules.vector.embedding import EmbeddingGenerator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class DenseRetrieverConfig:
    """Configuration for Dense Retriever"""
    model_name_or_path: str="."
    dim: int = 768
    index_path: str = None
    batch_size: int = 32
    api_key: str = None
    base_url: str = None
    embedding_model_name: str = None

    def validate(self):
        """Validate configuration parameters"""
        if not isinstance(self.model_name_or_path, str):
            raise ValueError("Model name must be a non-empty string.")
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if self.index_path and not isinstance(self.index_path, str):
            raise ValueError("Index directory path must be a string.")
        print("Dense configuration is valid.")


class DenseRetriever:
    """Dense Retriever for efficient document search using various embedding models"""

    def __init__(self, config: DenseRetrieverConfig, embedding_generator: EmbeddingGenerator):
        """
        Initialize the retriever.

        Args:
            config: DenseRetrieverConfig object containing configuration parameters
            embedding_generator: Instance of EmbeddingGenerator for creating embeddings
        """
        self.config = config
        self.config.validate()
        self.embedding_generator = embedding_generator

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(config.dim)
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.num_documents: int = 0

    def load_index(self, index_path: str = None):
        """Load the FAISS index and documents from disk"""
        if index_path is None:
            index_path = self.config.index_path

        try:
            # Load document data
            data = np.load(os.path.join(index_path, 'document.vecstore.npz'), allow_pickle=True)
            self.documents, self.embeddings = data['documents'].tolist(), data['embeddings'].tolist()

            # Load FAISS index
            self.index = faiss.read_index(os.path.join(index_path, 'faiss.index'))
            print(f"Index loaded successfully from {index_path}")

            # Cleanup
            del data
            gc.collect()

        except Exception as e:
            raise RuntimeError(f"Failed to load index from {index_path}: {str(e)}")

    def save_index(self, index_path: str = None):
        """Save the FAISS index and documents to disk"""
        if not self.index or not self.embeddings or not self.documents:
            raise ValueError("No index data to save")

        if index_path is None:
            index_path = self.config.index_path

        try:
            # Create directory if needed
            os.makedirs(index_path, exist_ok=True)
            print(f"Saving index to: {index_path}")

            # Save document data
            np.savez(
                os.path.join(index_path, 'document.vecstore'),
                embeddings=self.embeddings,
                documents=self.documents
            )

            # Save FAISS index
            faiss.write_index(self.index, os.path.join(index_path, 'faiss.index'))
            print(f"Index saved successfully to {index_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save index to {index_path}: {str(e)}")

    def add_texts(self, texts: List[str]):
        """
        Add multiple texts to the index.

        Args:
            texts: List of texts to add
        """
        # Handle empty texts
        texts = [text if text else "Empty document" for text in texts]

        # Generate embeddings using the embedding generator
        embeddings = self.embedding_generator.generate_embeddings(texts)
        # print(embeddings.shape)
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Update internal storage
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)
        self.num_documents += len(texts)

    def add_text(self, text: str):
        """Add a single text to the index"""
        self.add_texts([text])

    def build_from_texts(self, corpus: List[str]):
        """
        Process and index a list of texts in batches.

        Args:
            corpus: List of texts to index
        """
        if not corpus:
            return

        for i in tqdm(range(0, len(corpus), self.config.batch_size), desc="Building index"):
            batch = corpus[i:i + self.config.batch_size]
            self.add_texts(batch)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Retrieve the top_k documents relevant to the query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing retrieved documents and their scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query).astype('float32').reshape(1, -1)

        # Search index
        scores, indices = self.index.search(query_embedding, top_k)

        # Create results
        results = [
            {'text': self.documents[idx], 'score': score}
            for idx, score in zip(indices[0], scores[0])
        ]

        return results


# Example usage:
"""
# Initialize with OpenAI embeddings
config = DenseRetrieverConfig(
    model_name_or_path="text-embedding-3-large",
    dim=3072,
    index_path="path/to/index"
)
embedding_generator = OpenAIEmbedding(api_key="your-key", base_url="your-url", model=config.model_name_or_path)
retriever = DenseRetriever(config, embedding_generator)

# Initialize with FlagModel embeddings
config = DenseRetrieverConfig(
    model_name="BAAI/bge-base-en-v1.5",
    dim=768,
    index_path="path/to/index"
)
embedding_generator = FlagModelEmbedding(model_name=config.model_name)
retriever = DenseRetriever(config, embedding_generator)

# Use the retriever
texts = ["document1", "document2", "document3"]
retriever.build_from_texts(texts)
results = retriever.retrieve("query", top_k=2)
"""