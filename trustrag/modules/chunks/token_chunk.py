from typing import List, Callable, Optional, Union

import jieba
from transformers import AutoTokenizer

from trustrag.modules.chunks.base import BaseChunker
from trustrag.modules.document.rag_tokenizer import RagTokenizer


class TokenChunker(BaseChunker):
    def __init__(
            self,
            tokenizer_type: str = "rag",
            model_name_or_path: Optional[str] = None
    ):
        """
        Initialize the TokenChunker.

        :param chunk_size: The number of tokens per chunk, default is 64.
        :param tokenizer_type: The type of tokenizer, supports "rag", "jieba", and "hf", default is "rag".
        :param model_name_or_path: When tokenizer_type is "hf", specify the model name or path for Hugging Face.
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

    def get_chunks(self, paragraphs: Union[str, List[str]], chunk_size: int = 64) -> List[str]:
        """
        Split paragraphs into chunks of the specified size.

        :param paragraphs: A list of input paragraphs.
        :param chunk_size: The chunk size
        :return chunks: A list of chunks after splitting.
        """
        paragraphs = self.check_validation(paragraphs=paragraphs, chunk_size=chunk_size)
        chunks = []
        for paragraph in paragraphs:
            tokens = self.tokenizer_func(paragraph)
            chunks.extend([
                "".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)
            ])
        return chunks


if __name__ == '__main__':
    # with open("../../../data/docs/news.txt","r",encoding="utf-8") as f:
    with open("../../../data/docs/伊朗总统罹难事件.txt", "r", encoding="utf-8") as f:
        content = f.read()
    # print(content)
    tc = TokenChunker(tokenizer_type="jieba")
    chunks = tc.get_chunks([content],chunk_size=128)
    for chunk in chunks:
        print(f"Chunk Content：\n{chunk}")
