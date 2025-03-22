from typing import List, Union
from trustrag.modules.chunks.base import BaseChunker


class CharChunker(BaseChunker):
    """
    A character-based chunker that splits input texts into fixed-size chunks of characters.

    This class inherits from `BaseChunker` and implements the `get_chunks` method to divide
    input texts into smaller chunks, where each chunk contains a specified number of characters.
    This is useful for processing long texts in smaller, manageable pieces.

    """

    def __init__(self) -> None:
        """
        Initializes the CharChunker with a specified chunk size.

        """
        super().__init__()

    def get_chunks(self, paragraphs: Union[str, List[str]], chunk_size: int = 64) -> List[str]:
        """
        Splits the input paragraphs into chunks of characters based on the specified chunk size.

        Args:
            paragraphs (Union[str, List[str]]): A string or a list of strings (paragraphs) to be chunked.
            chunk_size (int): The size of each chunk.

        Returns:
            List[str]: A list of chunks, where each chunk is a string of characters.

        Raises:
            ValueError: If chunk_size is not a positive integer.
            TypeError: If paragraphs is not a string or a list of strings.
        """
        # check input valid
        paragraphs=self.check_validation(paragraphs=paragraphs, chunk_size=chunk_size)
        # use list comprehension to optimize performance
        chunks = [
            paragraph[i:i + chunk_size]
            for paragraph in paragraphs
            for i in range(0, len(paragraph), chunk_size)
        ]
        return chunks

if __name__ == "__main__":
    cc = CharChunker()
    with open("../../../data/docs/伊朗总统罹难事件.txt","r",encoding="utf-8") as f:
        content=f.read()
    print(cc.get_chunks([content],chunk_size=128))
    print(cc.get_chunks(content,chunk_size=128))
