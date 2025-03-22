from typing import List, Dict, Any, Union, Optional
import numpy as np
from trustrag.modules.chunks.base import BaseChunker
from trustrag.modules.retrieval.embedding import EmbeddingGenerator


class SemanticChunker(BaseChunker):
    """
    A class for semantically chunking text based on sentence embeddings.

    This chunker uses semantic similarity between consecutive sentences to identify
    natural breakpoints in the text, creating chunks that maintain semantic coherence.
    It supports multiple methods for determining where to break text into chunks based on
    similarity thresholds.
    """

    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize the SemanticChunker.

        Args:
            embedding_generator (EmbeddingGenerator): An implementation of EmbeddingGenerator
                to generate embeddings for sentences.
        """
        super().__init__()
        self.embedding_generator = embedding_generator
        self.results = None

    def compute_breakpoints(
            self,
            similarities: List[float],
            method: str = "percentile",
            threshold: float = 90
    ) -> List[int]:
        """
        Computes chunking breakpoints based on similarity drops between consecutive sentences.

        Args:
            similarities: List of similarity scores between consecutive sentences.
            method: Method to determine breakpoints, options:
                - 'percentile': Breaks at points below a percentile threshold.
                - 'standard_deviation': Breaks at points below mean - (threshold * std_dev).
                - 'interquartile': Breaks at points below Q1 - 1.5 * IQR.
            threshold: Threshold value, meaning depends on the method:
                - For 'percentile': The percentile below which to break (0-100).
                - For 'standard_deviation': Number of standard deviations below mean.
                - For 'interquartile': Not used directly, fixed at 1.5 * IQR.

        Returns:
            List of indices where chunk splits should occur.

        Raises:
            ValueError: If an invalid method is provided.
        """
        if not similarities:
            return []

        # Determine threshold based on selected method
        if method == "percentile":
            # Calculate X percentile of similarity scores
            threshold_value = np.percentile(similarities, threshold)
        elif method == "standard_deviation":
            # Calculate mean and standard deviation of similarity scores
            mean = np.mean(similarities)
            std_dev = np.std(similarities)
            # Set threshold as mean minus X standard deviations
            threshold_value = mean - (threshold * std_dev)
        elif method == "interquartile":
            # Calculate first and third quartiles (Q1 and Q3)
            q1, q3 = np.percentile(similarities, [25, 75])
            # Set threshold using IQR rule for outliers
            threshold_value = q1 - 1.5 * (q3 - q1)
        else:
            # Raise error if invalid method is provided
            raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

        # Return indices where similarity is below threshold
        return [i for i, sim in enumerate(similarities) if sim < threshold_value]

    def split_into_chunks(self, sentences: List[str], breakpoints: List[int]) -> List[str]:
        """
        Splits sentences into semantic chunks based on the identified breakpoints.

        Args:
            sentences: List of sentences to be chunked.
            breakpoints: Indices where chunking should occur.

        Returns:
            List of text chunks, with sentences joined by periods.
        """
        if not sentences:
            return []

        if not breakpoints:
            return [". ".join(sentences) + "."]

        chunks = []
        start = 0

        # Ensure breakpoints are in ascending order
        sorted_breakpoints = sorted(breakpoints)

        # Create chunks using the breakpoints
        for bp in sorted_breakpoints:
            if bp < len(sentences) - 1:  # Ensure breakpoint is valid
                chunks.append(". ".join(sentences[start:bp + 1]) + ".")
                start = bp + 1

        # Add remaining sentences as the last chunk
        if start < len(sentences):
            chunks.append(". ".join(sentences[start:]) + ".")

        return chunks

    def get_chunks(
            self,
            text: str,
            chunk_method: str = "percentile",
            threshold: float = 90
    ) -> List[str]:
        """
        Process a text string and split it into semantic chunks.

        Args:
            text (str): Input text to process.
            chunk_method (str): Method for determining breakpoints:
                - 'percentile': Use percentile-based thresholding.
                - 'standard_deviation': Use standard deviation-based thresholding.
                - 'interquartile': Use interquartile range for thresholding.
            threshold (float): Threshold value for the chosen method.

        Returns:
            List[str]: List of text chunks.
        """
        # Split the text into sentences
        sentences = text.split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text]

        # Generate embeddings for each sentence
        embeddings = self.embedding_generator.generate_embeddings(sentences)

        # Compute similarity between consecutive sentences
        similarities = [
            self.embedding_generator.cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Compute breakpoints using the specified method
        breakpoints = self.compute_breakpoints(
            similarities, method=chunk_method, threshold=threshold
        )

        # Split the text into chunks based on the breakpoints
        chunks = self.split_into_chunks(sentences, breakpoints)

        # Store the results for later reference
        self.results = {
            "sentences": sentences,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "breakpoints": breakpoints,
            "similarities": similarities
        }

        return chunks

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the last chunking operation.

        Returns:
            A dictionary containing chunking results or None if no chunking has been performed.
            The dictionary includes:
            - sentences: The original list of sentences
            - chunks: The resulting text chunks
            - num_chunks: Number of chunks created
            - breakpoints: Indices where text was split
            - similarities: Similarity scores between consecutive sentences
        """
        return self.results


class SentenceTransformerEmbedding(EmbeddingGenerator):
    """
    Embedding generator using Sentence Transformers models.

    This class implements the EmbeddingGenerator interface using the sentence-transformers
    library to generate sentence embeddings.
    """

    def __init__(
            self,
            model_name_or_path: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
            device: Optional[str] = None
    ):
        """
        Initialize the SentenceTransformerEmbedding.

        Args:
            model_name_or_path (str): The name or path of the sentence-transformers model.
                Default is "sentence-transformers/multi-qa-mpnet-base-cos-v1".
            device (Optional[str]): The device to use for computation ('cuda', 'cpu').
                If None, automatically uses CUDA if available.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self.embedding_size = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): A list of text strings to encode.

        Returns:
            np.ndarray: A 2D numpy array of shape (len(texts), embedding_size)
                containing the embeddings for each text.
        """
        return self.model.encode(texts, show_progress_bar=False)

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding vector.
            embedding2 (np.ndarray): Second embedding vector.

        Returns:
            float: Cosine similarity value between the two embeddings (-1 to 1).
        """
        from numpy.linalg import norm

        # Calculate cosine similarity
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))