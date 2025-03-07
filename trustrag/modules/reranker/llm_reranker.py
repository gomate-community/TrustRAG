#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: bge_reranker.py
@time: 2024/06/05
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trustrag.modules.reranker.base import BaseReranker

SETWISE_LIKELYHOOD = """
你是一个专业的搜索算法助手，可以根据找出与查询最相关的段落。
以下是{num}段文字，每段都用字母标识符依次表示。你需要根据与查询的相关性找出最相关的段落。
查询内容为：<查询>{query}</查询>
{docs}
根据与搜索查询相关性找出上述{num}段文字中最相关的段落。你应该只输出最相关段落的标识符。只回复最终结果，不要说任何其他话。请注意，如果有多个段落与查询相关度相同，则随机选择一个。
"""


class LLMRerankerConfig:
    """
    Configuration class for setting up a LLM reranker.

    Attributes:
        model_name_or_path (str): Path or model identifier for the pretrained model from Hugging Face's model hub.
        device (str): Device to load the model onto ('cuda' or 'cpu').
        api_key (str): API key for the reranker service.
        url (str): URL for the reranker service.
    """

    def __init__(
        self, model_name_or_path="Qwen2.5-7B-Instruct", api_key=None, url=None
    ):
        self.model_name_or_path = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.api_key = api_key
        self.url = url

    def log_config(self):
        # Log the current configuration settings
        return f"""
        LLMRerankerConfig:
            Model Name or Path: {self.model_name_or_path}
            Device: {self.device}
            URL: {self.url}
            API Key: {'*' * 8 if self.api_key else 'Not Set'}
        """


class LLMReranker(BaseReranker):
    """
    A reranker that utilizes a LLM to rerank a list of documents based on their relevance to a given query.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.CHARACTERS = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
        ]
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.rerank_model = (
            AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
            .half()
            .to(config.device)
            .eval()
        )
        self.device = config.device
        self.decoder_input_ids = self.rerank_tokenizer.encode(
            "<pad> 最相关的段落是：", return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        self.target_token_ids = self.rerank_tokenizer.batch_encode_plus(
            [
                f"<pad> 最相关的段落是：{self.CHARACTERS[i]}"
                for i in range(len(self.CHARACTERS))
            ],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).input_ids[:, -1]
        print("Successful load rerank model")

    def rerank(
        self,
        query: str,
        documents: List[str],
        k: int = 5,
        is_sorted: bool = True,
        method: str = "setwise_likelyhood",
    ) -> list[dict[str, Any]]:
        # Process input documents for uniqueness and formatting

        if method == "setwise_likelyhood":
            indexed_docs = "\n".join(
                [f"{self.CHARACTERS[i]}: {doc}" for i, doc in enumerate(documents)]
            )
            params = {"query": query, "docs": indexed_docs, "num": len(documents)}
            if len(documents) > 20:
                raise ValueError("目前暂不支持超过20条文档排序！")
            input_text = SETWISE_LIKELYHOOD.format(**params)
            input_ids = self.rerank_tokenizer(
                input_text, return_tensors="pt"
            ).input_ids.to(self.device)
            # Tokenize and predict relevance scores
            with torch.no_grad():

                logits = self.rerank_model(
                    input_ids=input_ids, decoder_input_ids=self.decoder_input_ids
                ).logits[0][-1]
                distributions = torch.softmax(logits, dim=0)
                scores = distributions[self.target_token_ids[: len(documents)]]

            # Pair documents with their scores, sort by scores in descending order
            if is_sorted:
                ranked_docs = sorted(
                    zip(documents, scores), key=lambda x: x[1], reverse=True
                )
                # Return the top k documents
                top_docs = [
                    {"text": doc, "score": score.item()} for doc, score in ranked_docs
                ]
            else:
                top_docs = [
                    {"text": doc, "score": score.item()}
                    for doc, score in zip(documents, scores)
                ]

            return top_docs
        else:
            raise NotImplementedError
