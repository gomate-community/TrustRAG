from typing import List, Dict, Any, Union, Optional
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from trustrag.modules.chunks.base import BaseChunker
from trustrag.modules.vector.embedding import EmbeddingGenerator,SentenceTransformerEmbedding


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
        Initialize the SemanticChunker with an embedding generator.

        Args:
            embedding_generator (EmbeddingGenerator): An embedding generator to create
                sentence embeddings for similarity comparison.
        """
        self.embedding_generator = embedding_generator

    def _split_text_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences using regex.

        Args:
            text (str): The input text to be split into sentences.

        Returns:
            List[str]: A list of individual sentences from the text.
        """
        # Split on common sentence delimiters (period, question mark, exclamation mark, semicolon, newline)
        sentences = re.split(r'[。；？！\n]+', text)
        # Filter out empty sentences
        return [sent for sent in sentences if sent.strip()]

    def _combine_sentences_with_context(self, sentences: List[Dict], buffer_size: int = 1) -> List[Dict]:
        """
        Combine each sentence with its surrounding context based on buffer size.

        Args:
            sentences (List[Dict]): List of sentence dictionaries with 'sentence' and 'index' keys.
            buffer_size (int): Number of sentences to include before and after the current sentence.

        Returns:
            List[Dict]: Updated sentences with 'combined_sentence' field added.
        """
        for i in range(len(sentences)):
            combined_sentence = ''

            # Add sentences before current sentence (based on buffer size)
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after current sentence (based on buffer size)
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Store the combined sentence
            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def _calculate_cosine_distances(self, sentences: List[Dict]) -> List[float]:
        """
        Calculate cosine distances between consecutive sentence embeddings.

        Args:
            sentences (List[Dict]): List of sentence dictionaries with 'combined_sentence' fields.

        Returns:
            List[float]: List of cosine distances between consecutive sentences.
        """
        # Generate embeddings for all combined sentences
        embeddings = self.embedding_generator.generate_embeddings(
            [s['combined_sentence'] for s in sentences]
        )

        # Store embeddings in the sentence dictionaries
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        # Calculate distances between consecutive sentences
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

            # Convert to cosine distance (1 - similarity)
            distance = 1 - similarity

            # Store distance
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance

        return distances

    def _find_breakpoints(self, distances: List[float], method: str, threshold: float) -> List[int]:
        """
        Find breakpoints in the text based on sentence distances.

        Args:
            distances (List[float]): List of cosine distances between consecutive sentences.
            method (str): Method to determine breakpoints ('percentile', 'absolute', or 'dynamic').
            threshold (float): Threshold value for the chosen method.

        Returns:
            List[int]: Indices of sentences that mark the end of a chunk.
        """
        if method == "percentile":
            # Use percentile of distances as threshold
            breakpoint_distance_threshold = np.percentile(distances, threshold)
            indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        elif method == "absolute":
            # Use absolute threshold value
            indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold]

        elif method == "dynamic":
            # Dynamic thresholding based on local context
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            # Threshold is mean + (threshold * standard deviation)
            dynamic_threshold = mean_distance + (threshold * std_distance)
            indices_above_thresh = [i for i, x in enumerate(distances) if x > dynamic_threshold]

        else:
            raise ValueError(f"Unknown chunk method: {method}. Choose from 'percentile', 'absolute', or 'dynamic'")

        return indices_above_thresh

    def _create_chunks_from_breakpoints(self, sentences: List[Dict], breakpoints: List[int]) -> List[str]:
        """
        Create text chunks based on the identified breakpoints.

        Args:
            sentences (List[Dict]): List of sentence dictionaries.
            breakpoints (List[int]): Indices marking the end of chunks.

        Returns:
            List[str]: List of text chunks.
        """
        chunks = []
        start_index = 0

        # Create chunks based on breakpoints
        for index in breakpoints:
            end_index = index + 1  # Include the sentence at the breakpoint
            group = sentences[start_index:end_index]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = end_index

        # Add the final chunk if there are remaining sentences
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

    def get_chunks(
            self,
            text: str,
            chunk_method: str = "percentile",
            threshold: float = 90,
            buffer_size: int = 1
    ) -> List[str]:
        """
        Split text into semantically coherent chunks based on embedding similarity.

        Args:
            text (str): The input text to split into chunks.
            chunk_method (str): Method to determine chunk breakpoints:
                - 'percentile': Use a percentile threshold of distance distribution
                - 'absolute': Use an absolute distance threshold
                - 'dynamic': Use mean + (threshold * std_dev) as the threshold
            threshold (float): Threshold value for the chosen method:
                - For 'percentile': A percentile value (0-100)
                - For 'absolute': A direct cosine distance threshold (0-1)
                - For 'dynamic': A multiplier for standard deviation
            buffer_size (int): Number of sentences to include as context when calculating similarity

        Returns:
            List[str]: List of text chunks divided at semantic breakpoints
        """
        # Split text into sentences
        single_sentences = self._split_text_into_sentences(text)

        # Create sentence dictionaries with indices
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences)]

        # Combine sentences with context for better semantic representation
        sentences = self._combine_sentences_with_context(sentences, buffer_size)

        # Calculate cosine distances between consecutive sentences
        distances = self._calculate_cosine_distances(sentences)

        # Find breakpoints based on the specified method and threshold
        breakpoints = self._find_breakpoints(distances, chunk_method, threshold)

        # Create chunks based on breakpoints
        chunks = self._create_chunks_from_breakpoints(sentences, breakpoints)

        return chunks




# Now use the SemanticChunker with the example file
if __name__ == "__main__":
    # Load the text from the file
    with open('../../../data/docs/新浪新闻.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Create the embedding generator
    # If you're using a local model like in your original code, specify the path
    embedding_generator = SentenceTransformerEmbedding(
        model_name_or_path="G:/pretrained_models/mteb/bge-m3"  # Change to your model path if needed
    )


    chunker = SemanticChunker(embedding_generator)

    # Get chunks using different methods

    # Method 1: Using percentile thresholding (default)
    percentile_chunks = chunker.get_chunks(
        text=text,
        chunk_method="percentile",
        threshold=90,  # 90th percentile of distances
        buffer_size=1
    )

    # Method 2: Using absolute thresholding
    absolute_chunks = chunker.get_chunks(
        text=text,
        chunk_method="absolute",
        threshold=0.2,  # Cosine distance threshold of 0.2
        buffer_size=1
    )

    # Method 3: Using dynamic thresholding
    dynamic_chunks = chunker.get_chunks(
        text=text,
        chunk_method="dynamic",
        threshold=1.5,  # 1.5 standard deviations above the mean
        buffer_size=1
    )

    # Print the results
    print(f"Text was divided into {len(percentile_chunks)} chunks using percentile method")
    for i, chunk in enumerate(percentile_chunks):
        print(f"Chunk #{i + 1} ({len(chunk)} chars)")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print()

    # You can also print or process the chunks from other methods
    print(f"\nNumber of chunks using absolute threshold: {len(absolute_chunks)}")
    print(f"Number of chunks using dynamic threshold: {len(dynamic_chunks)}")