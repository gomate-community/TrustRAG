import re
from typing import List, Callable, Optional,Union

import jieba
from transformers import AutoTokenizer
from trustrag.modules.chunks.base import BaseChunker
from trustrag.modules.document.rag_tokenizer import RagTokenizer


class SentenceChunker(BaseChunker):
    """
    A class for splitting text into chunks based on sentences, ensuring each chunk does not exceed a specified token size.

    This class handles both Chinese and English text, splitting it into sentences using punctuation marks.
    It groups these sentences into chunks while ensuring the token count in each chunk stays within
    the specified `chunk_size` limit.

    # Examples:
    #     >>> chunker = SentenceChunker(chunk_size=128)
    #     >>> text = "这是第一句。这是第二句！How are you? I'm fine."
    #     >>> chunks = chunker.get_chunks([text])
    #     >>> print(chunks)
    #     ["这是第一句。这是第二句！How are you? I'm fine."]

    Attributes:
        tokenizer: A tokenizer function used to count tokens in sentences.
        chunk_size (int): The maximum number of tokens allowed per chunk.
    """

    def __init__(
            self,
            tokenizer_type: str = "rag",
            model_name_or_path: Optional[str] = None
    ):
        """
        Initialize the SentenceChunker with a specified chunk size.

        Args:
        """
        super().__init__()
        self.tokenizer_func = self.init_tokenizer(tokenizer_type, model_name_or_path)

    def init_tokenizer(
            self,
            tokenizer_type: str = "rag",
            model_name_or_path: Optional[str] = None
    ) -> Callable[[str], List[str]]:
        """
        Initialize the tokenizer.

        :param tokenizer_type: The type of tokenizer, supports "rag", "jieba", and "hf".
        :param model_name_or_path: When tokenizer_type is "hf", specify the model name or path for Hugging Face.
        :return: A tokenizer function that takes a string as input and returns a list of tokens.
        """
        if tokenizer_type == "rag":
            return RagTokenizer().tokenize
        elif tokenizer_type == "jieba":
            return lambda text: list(jieba.cut(text))
        elif tokenizer_type == "hf":
            if model_name_or_path is None:
                model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"  # Default model
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            return lambda text: [tokenizer.decode([token]) for token in tokenizer.encode(text)]
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    def split_sentences(self, text: str) -> List[str]:
        """
        Split the input text into sentences based on Chinese and English punctuation marks.

        Args:
            text: The input text to be split into sentences.

        Returns:
            A list of sentences extracted from the input text.
        """
        if not text or not text.strip():
            return []

        # Improved regex for sentence splitting - handles more punctuation types
        sentence_endings = re.compile(r'([。！？.!?]+)')
        sentences = sentence_endings.split(text)

        # Merge punctuation marks with their preceding sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                if sentences[i]:
                    result.append(sentences[i] + sentences[i + 1])

        # Handle the last sentence if it lacks punctuation
        if len(sentences) % 2 == 1 and sentences[-1]:
            result.append(sentences[-1])

        # Remove whitespace and filter out empty sentences
        result = [sentence.strip() for sentence in result if sentence.strip()]

        return result

    def process_text_chunks(self, chunks: List[str]) -> List[str]:
        """
        Preprocess text chunks by normalizing excessive newlines and spaces.

        Args:
            chunks: A list of text chunks to be processed.

        Returns:
            A list of processed text chunks with normalized formatting.
        """
        if not chunks:
            return []

        processed_chunks = []
        for chunk in chunks:
            # Use regex for more efficient normalization
            # Normalize multiple consecutive newlines to double newlines
            chunk = re.sub(r'\n{3,}', '\n\n', chunk)

            # Normalize multiple consecutive spaces to double spaces
            chunk = re.sub(r' {3,}', '  ', chunk)

            processed_chunks.append(chunk)

        return processed_chunks

    def split_large_sentence(self, sentence: str,chunk_size:int) -> List[str]:
        """
        Split a large sentence that exceeds the chunk size into smaller parts.

        Args:
            sentence: The sentence to be split.
            chunk_size:

        Returns:
            A list of smaller sentence parts that fit within the token limit.
        """
        tokens = self.tokenizer_func(sentence)

        # If the sentence is already within the limit, return it as is
        if len(tokens) <= chunk_size:
            return [sentence]

        # Split the sentence into smaller parts based on token count
        sentence_parts = []
        current_part = []
        current_tokens = 0

        for token in tokens:
            # Check if adding the current token would exceed the chunk size
            if current_tokens + 1 > chunk_size:
                # Add the current part to the list of parts and reset
                if current_part:
                    part_text = "".join(current_part)
                    sentence_parts.append(part_text)
                    current_part = []
                    current_tokens = 0

            # Add the current token to the current part
            current_part.append(token)
            current_tokens += 1

        # Add the last part if it contains any tokens
        if current_part:
            part_text = "".join(current_part)
            sentence_parts.append(part_text)

        return sentence_parts

    def get_chunks(self, paragraphs: Union[str, List[str]], chunk_size: int = 64) -> List[str]:
        """
        Split a list of paragraphs into chunks based on the specified token size.

        Args:
            paragraphs: A list of paragraphs to be chunked.
            chunk_size: Optional. The maximum number of tokens allowed per chunk.
                        If not provided, the instance's chunk_size will be used.

        Returns:
            A list of text chunks, each containing sentences that fit within the token limit.
        """
        paragraphs=self.check_validation(paragraphs,chunk_size)
        if not paragraphs:
            return []

        # Combine paragraphs into a single text
        text = ''.join(paragraphs)

        # Split the text into sentences
        sentences = self.split_sentences(text)

        # If no sentences are found, treat paragraphs as sentences
        if not sentences:
            sentences = [p for p in paragraphs if p.strip()]
            if not sentences:
                return []

        chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        # Iterate through sentences and build chunks based on token count
        for sentence in sentences:
            tokens = self.tokenizer_func(sentence)
            # Check if the current sentence exceeds the chunk size
            if len(tokens) > chunk_size:
                # If we have content in the current chunk, finalize it
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_chunk_tokens = 0

                # Split the large sentence into smaller parts
                sentence_parts = self.split_large_sentence(sentence,chunk_size)

                # Add each part as a separate chunk or to the current chunk
                for part in sentence_parts:
                    part_tokens = self.tokenizer_func(part)

                    # Check if this part can be added to the current chunk
                    if current_chunk_tokens + len(part_tokens) <= chunk_size:
                        current_chunk.append(part)
                        current_chunk_tokens += len(part_tokens)
                    else:
                        # Finalize the current chunk if not empty
                        if current_chunk:
                            chunks.append(''.join(current_chunk))
                            current_chunk = []
                            current_chunk_tokens = 0

                        # If the part fits in a chunk by itself, add it directly
                        if len(part_tokens) <= chunk_size:
                            current_chunk.append(part)
                            current_chunk_tokens = len(part_tokens)
                        else:
                            # This should not happen after split_large_sentence, but just to be safe
                            chunks.append(part)
            else:
                # Handle normal-sized sentences
                if current_chunk_tokens + len(tokens) <= chunk_size:
                    # Add sentence to the current chunk if it fits
                    current_chunk.append(sentence)
                    current_chunk_tokens += len(tokens)
                else:
                    # Finalize the current chunk and start a new one
                    if current_chunk:  # Check to avoid empty chunks
                        chunks.append(''.join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_tokens = len(tokens)

        # Add the last chunk if it contains any sentences
        if current_chunk:
            chunks.append(''.join(current_chunk))

        # Preprocess the chunks to normalize formatting
        chunks = self.process_text_chunks(chunks)
        return chunks


if __name__ == '__main__':
    # with open("../../../data/docs/伊朗总统罹难事件.txt", "r", encoding="utf-8") as f:
    with open("../../../data/docs/bbc新闻.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # Create a SentenceChunker with a chunk size of 128 tokens
    chunker = SentenceChunker(tokenizer_type="rag")

    # Generate chunks from the content
    chunks = chunker.get_chunks([content],chunk_size=128)

    # Print each chunk with a separator for clarity
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:\n length:{len(chunk)} \n{chunk}")
