import re
from sklearn.metrics.pairwise import cosine_similarity
from trustrag.modules.document import rag_tokenizer
from trustrag.modules.chunks.base import BaseChunker
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

class SemanticChunker(BaseChunker):
    """
    A class for splitting text into chunks based on semantic similarity between sentences.

    This class uses sentence embeddings to calculate the semantic similarity between sentences
    and groups them into chunks based on a similarity threshold. It ensures that each chunk
    contains semantically related sentences.

    Attributes:
        tokenizer (callable): A tokenizer function used to count tokens in sentences.
        chunk_size (int): The maximum number of tokens allowed per chunk.
        similarity_threshold (float): The threshold for semantic similarity to group sentences.
        embeddings_model: The embedding model used to generate sentence embeddings.
                          Can be either OpenAIEmbeddings or SentenceTransformer.
    """

    def __init__(self, chunk_size=512, similarity_threshold=0.8, embedding_model="sentence-transformers", model_name="all-MiniLM-L6-v2"):
        """
        Initializes the SemanticChunker with a tokenizer, chunk size, similarity threshold, and embedding model.

        Args:
            chunk_size (int, optional): The maximum number of tokens allowed per chunk. Defaults to 512.
            similarity_threshold (float, optional): The threshold for semantic similarity to group sentences. Defaults to 0.8.
            embedding_model (str, optional): The embedding model to use. Options: "sentence-transformers" or "openai". Defaults to "sentence-transformers".
            model_name (str, optional): The name of the model to use. For "sentence-transformers", it's the model name (e.g., "all-MiniLM-L6-v2").
                                        For "openai", it's the model name (e.g., "text-embedding-ada-002"). Defaults to "all-MiniLM-L6-v2".
        """
        super().__init__()
        self.tokenizer = rag_tokenizer
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

        if embedding_model == "sentence-transformers":
            self.embeddings_model = SentenceTransformer(model_name)
        elif embedding_model == "openai":
            self.embeddings_model = OpenAIEmbeddings(model=model_name)
        else:
            raise ValueError("Invalid embedding_model. Choose 'sentence-transformers' or 'openai'.")

    def split_sentences(self, text: str) -> list[str]:
        """
        Splits the input text into sentences based on Chinese and English punctuation marks.

        Args:
            text (str): The input text to be split into sentences.

        Returns:
            list[str]: A list of sentences extracted from the input text.
        """
        # Use regex to split text by sentence-ending punctuation marks
        sentence_endings = re.compile(r'([。！？.!?])')
        sentences = sentence_endings.split(text)

        # Merge punctuation marks with their preceding sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if sentences[i]:
                result.append(sentences[i] + sentences[i + 1])

        # Handle the last sentence if it lacks punctuation
        if sentences[-1]:
            result.append(sentences[-1])

        # Remove whitespace and filter out empty sentences
        result = [sentence.strip() for sentence in result if sentence.strip()]

        return result

    def get_sentence_embeddings(self, sentences: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of sentences using the selected embedding model.

        Args:
            sentences (list[str]): A list of sentences to generate embeddings for.

        Returns:
            list[list[float]]: A list of sentence embeddings.
        """
        if isinstance(self.embeddings_model, SentenceTransformer):
            return self.embeddings_model.encode(sentences)
        elif isinstance(self.embeddings_model, OpenAIEmbeddings):
            return self.embeddings_model.embed_documents(sentences)
        else:
            raise ValueError("Unsupported embedding model.")

    def calculate_cosine_distances(self, embeddings: list[list[float]]) -> list[float]:
        """
        Calculates the cosine distances between consecutive sentence embeddings.

        Args:
            embeddings (list[list[float]]): A list of sentence embeddings.

        Returns:
            list[float]: A list of cosine distances between consecutive sentences.
        """
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def get_chunks(self, paragraphs: list[str]) -> list[str]:
        """
        Splits a list of paragraphs into chunks based on semantic similarity and token size.

        Args:
            paragraphs (list[str]|str): A list of paragraphs to be chunked.

        Returns:
            list[str]: A list of text chunks, each containing semantically related sentences.
        """
        # Combine paragraphs into a single text
        text = ''.join(paragraphs)

        # Split the text into sentences
        sentences = self.split_sentences(text)

        # If no sentences are found, treat paragraphs as sentences
        if len(sentences) == 0:
            sentences = paragraphs

        # Generate embeddings for sentences
        embeddings = self.get_sentence_embeddings(sentences)

        # Calculate cosine distances between consecutive sentences
        distances = self.calculate_cosine_distances(embeddings)

        # Determine breakpoints based on the similarity threshold
        breakpoint_indices = [i for i, distance in enumerate(distances) if distance > (1 - self.similarity_threshold)]
        print(breakpoint_indices)
        # Combine sentences into chunks
        chunks = []
        start_index = 0
        for index in breakpoint_indices:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join(group)
            chunks.append(combined_text)
            start_index = index + 1

        # Add the last chunk if it contains any sentences
        if start_index < len(sentences):
            combined_text = ' '.join(sentences[start_index:])
            chunks.append(combined_text)

        # Preprocess the chunks to normalize formatting
        chunks = self.process_text_chunks(chunks)
        return chunks

    def process_text_chunks(self, chunks: list[str]) -> list[str]:
        """
        Preprocesses text chunks by normalizing excessive newlines and spaces.

        Args:
            chunks (list[str]): A list of text chunks to be processed.

        Returns:
            list[str]: A list of processed text chunks with normalized formatting.
        """
        processed_chunks = []
        for chunk in chunks:
            # Normalize four or more consecutive newlines
            while '\n\n\n\n' in chunk:
                chunk = chunk.replace('\n\n\n\n', '\n\n')

            # Normalize four or more consecutive spaces
            while '    ' in chunk:
                chunk = chunk.replace('    ', '  ')

            processed_chunks.append(chunk)

        return processed_chunks

if __name__ == '__main__':
    with open("../../../data/docs/伊朗总统罹难事件.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # Example 1: Use SentenceTransformer
    sc_st = SemanticChunker(embedding_model="sentence-transformers", model_name="all-MiniLM-L6-v2")
    chunks_st = sc_st.get_chunks([content])
    for chunk in chunks_st:
        print(f"SentenceTransformer Chunk:\n{chunk}")

    # # Example 2: Use OpenAIEmbeddings
    # sc_openai = SemanticChunker(embedding_model="openai", model_name="text-embedding-ada-002")
    # chunks_openai = sc_openai.get_chunks([content])
    # for chunk in chunks_openai:
    #     print(f"OpenAIEmbeddings Chunk:\n{chunk}")