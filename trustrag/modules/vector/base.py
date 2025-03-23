
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional


class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.
    All embedding models should inherit from this class and implement the required methods.
    """
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to generate embeddings for
            
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dimension)
        """
        pass

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text string to generate embedding for
            
        Returns:
            np.ndarray: Embedding vector with shape (embedding_dimension,)
        """
        return self.generate_embeddings([text])[0]

    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1 (np.ndarray): First vector
            vector2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return 0 if not magnitude else dot_product / magnitude
