from typing import List, Union

class BaseChunker(object):
    def __init__(self):
        pass
    def get_chunks(self, paragraphs: Union[str, List[str]], chunk_size: int = 64):
        raise NotImplementedError

    @staticmethod
    def check_validation(paragraphs: Union[str, List[str]],chunk_size: int):
        # check input validation
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        elif not isinstance(paragraphs, list) or not all(isinstance(p, str) for p in paragraphs):
            raise TypeError("The 'paragraphs' parameter must be a string or a list of strings.")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("The 'chunk_size' parameter must be a positive integer.")
        return paragraphs