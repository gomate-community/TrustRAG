import numpy as np
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from trustrag.modules.chunks.semantic_chunk import SemanticChunker
from trustrag.modules.retrieval.embedding import SentenceTransformerEmbedding

def example_usage():
    """
    Example demonstrating how to use the SemanticChunker with SentenceTransformerEmbedding.
    """
    # Initialize the embedding generator
    embedding_generator = SentenceTransformerEmbedding(
        model_name_or_path="G:/pretrained_models/mteb/all-MiniLM-L6-v2"
    )

    # Initialize the semantic chunker
    chunker = SemanticChunker(embedding_generator=embedding_generator)

    # Example text with clear topic changes
    text = """
    Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

    气候变化真的大.
    
    The Python programming language is widely used in machine learning due to its simplicity and the availability of powerful libraries. Libraries such as TensorFlow, PyTorch, and scikit-learn provide tools for implementing machine learning algorithms efficiently. Python's readability makes it ideal for teaching and learning machine learning concepts. Its extensive ecosystem of data science tools also makes it suitable for production environments.
    """

    # Get chunks using different methods
    percentile_chunks = chunker.get_chunks(text, chunk_method="percentile", threshold=20)
    std_dev_chunks = chunker.get_chunks(text, chunk_method="standard_deviation", threshold=1.0)
    print(chunker.get_results())
    # Get the results
    results = chunker.get_results()

    # Print the chunks
    print(f"Number of chunks using percentile method: {len(percentile_chunks)}")
    for i, chunk in enumerate(percentile_chunks):
        print(f"/nChunk {i + 1}:")
        print(chunk)

    # Visualize the similarity scores and breakpoints
    visualize_similarities(results["similarities"], results["breakpoints"])


def visualize_similarities(similarities: List[float], breakpoints: List[int]):
    """
    Visualize the similarity scores between consecutive sentences and the detected breakpoints.

    Args:
        similarities: List of similarity scores between consecutive sentences
        breakpoints: List of indices where chunk splits occur
    """
    plt.figure(figsize=(12, 6))
    plt.plot(similarities, marker='o', linestyle='-', markersize=8)

    # Highlight breakpoints
    for bp in breakpoints:
        plt.axvline(x=bp, color='red', linestyle='--', alpha=0.7)
        plt.plot(bp, similarities[bp], 'ro', markersize=10)

    plt.title('Semantic Similarity Between Consecutive Sentences')
    plt.xlabel('Sentence Pair Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add horizontal line at mean similarity
    mean_sim = np.mean(similarities)
    plt.axhline(y=mean_sim, color='green', linestyle='--', alpha=0.7,
                label=f'Mean Similarity: {mean_sim:.3f}')

    plt.legend()
    plt.savefig('semantic_chunking_visualization.png')
    plt.show()


if __name__ == "__main__":
    example_usage()
