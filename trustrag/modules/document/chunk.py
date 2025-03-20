import re
from tqdm import tqdm
from trustrag.modules.document import rag_tokenizer


class TextChunker:
    def __init__(self):
        """
        Initializes the TextChunker with a tokenizer.
        """
        self.tokenizer = rag_tokenizer

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

    def split_large_sentence(self, sentence: str, chunk_size: int) -> list[str]:
        """
        Splits a large sentence that exceeds the chunk size into smaller parts.

        Args:
            sentence (str): The sentence to be split.
            chunk_size (int): The maximum number of tokens allowed per chunk.

        Returns:
            list[str]: A list of smaller sentence parts that fit within the token limit.
        """
        tokens = self.tokenizer.tokenize(sentence)

        # If the sentence is already within the limit, return it as is
        if len(tokens) <= chunk_size:
            return [sentence]

        # Split the sentence into smaller parts based on token count
        sentence_parts = []
        current_part = []
        current_tokens = 0

        for i, token in enumerate(tokens):

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

    def get_chunks(self, paragraphs: list[str], chunk_size: int) -> list[str]:
        """
        Splits a list of paragraphs into chunks based on a specified token size.

        Args:
            paragraphs (list[str]): A list of paragraphs to be chunked.
            chunk_size (int): The maximum number of tokens allowed per chunk.

        Returns:
            list[str]: A list of text chunks, each containing sentences that fit within the token limit.
        """
        # Combine paragraphs into a single text
        text = ''.join(paragraphs)

        # Split the text into sentences
        sentences = self.split_sentences(text)

        # If no sentences are found, treat paragraphs as sentences
        if len(sentences) == 0:
            sentences = paragraphs

        chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        # Iterate through sentences and build chunks based on token count
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)

            # Check if the current sentence exceeds the chunk size
            if len(tokens) > chunk_size:
                # If we have content in the current chunk, finalize it
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_chunk_tokens = 0

                # Split the large sentence into smaller parts
                sentence_parts = self.split_large_sentence(sentence, chunk_size)

                # Add each part as a separate chunk
                for part in sentence_parts:
                    part_tokens = self.tokenizer.tokenize(part)

                    # Check if this part can be added to the current chunk
                    if current_chunk_tokens + len(part_tokens) <= chunk_size:
                        current_chunk.append(part)
                        current_chunk_tokens += len(part_tokens)
                    else:
                        # Finalize the current chunk and add this part as a new chunk
                        if current_chunk:
                            chunks.append(''.join(current_chunk))
                        current_chunk = [part]
                        current_chunk_tokens = len(part_tokens)
            else:
                # Handle normal-sized sentences as before
                if current_chunk_tokens + len(tokens) <= chunk_size:
                    # Add sentence to the current chunk if it fits
                    current_chunk.append(sentence)
                    current_chunk_tokens += len(tokens)
                else:
                    # Finalize the current chunk and start a new one
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
    # Example usage
    paragraphs = ['Hello!\nHi!\nGoodbye!']
    tc = TextChunker()
    chunk_size = 512
    chunks = tc.get_chunks(paragraphs, chunk_size)

    # Print the resulting chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}\n")