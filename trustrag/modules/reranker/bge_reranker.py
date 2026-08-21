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
from typing import List, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from trustrag.modules.reranker.base import BaseReranker


class BgeRerankerConfig:
    """
    Configuration class for setting up a BERT-based reranker.

    Attributes:
        model_name_or_path (str): Path or model identifier for the pretrained model from Hugging Face's model hub.
        device (str): Device to load the model onto (for example, 'cuda', 'mps', or 'cpu').
        batch_size (int): Maximum number of query-document pairs per inference batch.
        max_length (int): Maximum tokenized sequence length.
        api_key (str): API key for the reranker service.
        url (str): URL for the reranker service.
    """

    def __init__(
        self,
        model_name_or_path='bert-base-uncased',
        api_key=None,
        url=None,
        device=None,
        batch_size=32,
        max_length=512,
    ):
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer")

        self.model_name_or_path = model_name_or_path
        self.device = device or self._default_device()
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid device: {self.device}") from exc
        self.batch_size = batch_size
        self.max_length = max_length
        self.api_key = api_key
        self.url = url

    @staticmethod
    def _default_device():
        if torch.cuda.is_available():
            return 'cuda'
        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is not None and mps_backend.is_available():
            return 'mps'
        return 'cpu'

    def log_config(self):
        # Log the current configuration settings
        return f"""
        BgeRerankerConfig:
            Model Name or Path: {self.model_name_or_path}
            Device: {self.device}
            Batch Size: {self.batch_size}
            Max Length: {self.max_length}
            URL: {self.url}
            API Key: {'*' * 8 if self.api_key else 'Not Set'}
        """


class BgeReranker(BaseReranker):
    """
    A reranker that utilizes a BERT-based model for sequence classification
    to rerank a list of documents based on their relevance to a given query.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path
        )
        if torch.device(config.device).type == 'cuda':
            self.rerank_model = self.rerank_model.half()
        self.rerank_model = self.rerank_model.to(config.device).eval()
        self.device = config.device
        print('Successful load rerank model')

    def rerank(
        self,
        query: str,
        documents: List[str],
        k: int = 5,
        is_sorted: bool = True,
    ) -> list[dict[str, Any]]:
        """Score documents and return the top-k results when sorting is enabled.

        Score-only mode preserves the input order and returns every document.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not documents:
            return []

        scores = []
        for start in range(0, len(documents), self.config.batch_size):
            batch_documents = documents[start:start + self.config.batch_size]
            pairs = [[query, document] for document in batch_documents]
            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.config.max_length,
            ).to(self.device)
            with torch.inference_mode():
                batch_scores = (
                    self.rerank_model(**inputs, return_dict=True)
                    .logits.view(-1)
                    .float()
                    .cpu()
                    .tolist()
                )
            scores.extend(batch_scores)

        if len(scores) != len(documents):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {len(documents)} documents"
            )

        # Pair documents with their scores, sort by scores in descending order
        if is_sorted:
            ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            # Return the top k documents
            top_docs = [{'text': doc, 'score': score} for doc, score in ranked_docs[:k]]
        else:
            top_docs = [{'text': doc, 'score': score} for doc, score in zip(documents, scores)]
        return top_docs
