"""Unit tests for the BGE reranker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from trustrag.modules.reranker.bge_reranker import BgeReranker


class FakeInputs(dict):
    """Minimal tokenizer output compatible with the reranker."""

    def to(self, device):
        """Record the target device and return this mapping."""
        self.device = device
        return self


def make_reranker(scores):
    """Build a reranker while bypassing pretrained model loading."""
    reranker = object.__new__(BgeReranker)
    reranker.rerank_tokenizer = Mock(
        return_value=FakeInputs(input_ids=torch.tensor([[1]]))
    )
    reranker.rerank_model = Mock(
        return_value=SimpleNamespace(logits=torch.tensor(scores))
    )
    reranker.device = "cpu"
    return reranker


def test_rerank_returns_only_the_highest_scoring_k_documents():
    """Sorted reranking must honor k after ordering by relevance."""
    reranker = make_reranker([0.1, 0.9, 0.4])

    results = reranker.rerank(
        query="query",
        documents=["low", "high", "medium"],
        k=2,
    )

    assert [result["text"] for result in results] == ["high", "medium"]
    assert [result["score"] for result in results] == pytest.approx([0.9, 0.4])


def test_rerank_preserves_input_order_when_sorting_is_disabled():
    """Score-only mode must preserve one result for every input document."""
    reranker = make_reranker([0.1, 0.9, 0.4])

    results = reranker.rerank(
        query="query",
        documents=["first", "second", "third"],
        k=1,
        is_sorted=False,
    )

    assert [result["text"] for result in results] == ["first", "second", "third"]


def test_rerank_returns_empty_result_without_running_the_model():
    """Empty input must not be passed to a tokenizer or model."""
    reranker = make_reranker([])

    assert reranker.rerank(query="query", documents=[]) == []
    reranker.rerank_tokenizer.assert_not_called()
    reranker.rerank_model.assert_not_called()


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_rerank_rejects_invalid_k(k):
    """Invalid result limits must fail before running model inference."""
    reranker = make_reranker([0.5])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        reranker.rerank(query="query", documents=["document"], k=k)

    reranker.rerank_tokenizer.assert_not_called()
    reranker.rerank_model.assert_not_called()
