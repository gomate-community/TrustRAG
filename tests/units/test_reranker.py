"""Unit tests for the BGE reranker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from trustrag.modules.reranker.bge_reranker import BgeReranker, BgeRerankerConfig


class FakeInputs(dict):
    """Minimal tokenizer output compatible with the reranker."""

    def to(self, device):
        """Record the target device and return this mapping."""
        self.device = device
        return self


def make_reranker(scores, batch_size=32, max_length=512):
    """Build a reranker while bypassing pretrained model loading."""
    score_cursor = 0

    def tokenize(pairs, **kwargs):
        return FakeInputs(input_ids=torch.ones((len(pairs), 1), dtype=torch.long))

    def run_model(input_ids, **kwargs):
        nonlocal score_cursor
        current_batch_size = input_ids.shape[0]
        batch_scores = scores[score_cursor:score_cursor + current_batch_size]
        score_cursor += current_batch_size
        return SimpleNamespace(logits=torch.tensor(batch_scores))

    reranker = object.__new__(BgeReranker)
    reranker.config = SimpleNamespace(
        batch_size=batch_size,
        max_length=max_length,
    )
    reranker.rerank_tokenizer = Mock(side_effect=tokenize)
    reranker.rerank_model = Mock(side_effect=run_model)
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


def test_rerank_batches_model_inference_and_forwards_max_length():
    """Large candidate sets must be split into configured inference batches."""
    reranker = make_reranker(
        scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        batch_size=2,
        max_length=128,
    )

    results = reranker.rerank(
        query="query",
        documents=["one", "two", "three", "four", "five"],
        is_sorted=False,
    )

    assert [result["score"] for result in results] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert [len(call.args[0]) for call in reranker.rerank_tokenizer.call_args_list] == [
        2,
        2,
        1,
    ]
    assert {
        call.kwargs["max_length"] for call in reranker.rerank_tokenizer.call_args_list
    } == {128}
    assert reranker.rerank_model.call_count == 3


def test_rerank_rejects_mismatched_model_output():
    """A model output shape mismatch must not silently drop documents."""
    reranker = make_reranker(scores=[0.5])

    with pytest.raises(
        RuntimeError,
        match="Reranker returned 1 scores for 2 documents",
    ):
        reranker.rerank(query="query", documents=["one", "two"])


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_rerank_rejects_invalid_k(k):
    """Invalid result limits must fail before running model inference."""
    reranker = make_reranker([0.5])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        reranker.rerank(query="query", documents=["document"], k=k)

    reranker.rerank_tokenizer.assert_not_called()
    reranker.rerank_model.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("batch_size", 0, "batch_size must be a positive integer"),
        ("batch_size", True, "batch_size must be a positive integer"),
        ("max_length", -1, "max_length must be a positive integer"),
        ("max_length", 1.5, "max_length must be a positive integer"),
        ("device", "not-a-device", "Invalid device"),
    ],
)
def test_config_rejects_invalid_inference_options(option, value, message):
    """Invalid inference settings must fail during configuration."""
    with pytest.raises(ValueError, match=message):
        BgeRerankerConfig(**{option: value})


def test_config_prefers_mps_when_cuda_is_unavailable(monkeypatch):
    """Automatic device selection must support Apple Silicon acceleration."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert BgeRerankerConfig().device == "mps"


@pytest.mark.parametrize(
    ("device", "uses_half_precision"),
    [("cpu", False), ("mps", False), ("cuda:0", True)],
)
def test_model_uses_half_precision_only_on_cuda(
    monkeypatch,
    device,
    uses_half_precision,
):
    """CPU and MPS inference must retain the model's safe default precision."""
    tokenizer = Mock()
    model = Mock()
    model.half.return_value = model
    model.to.return_value = model
    model.eval.return_value = model
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(
        "trustrag.modules.reranker.bge_reranker.AutoTokenizer.from_pretrained",
        tokenizer_loader,
    )
    monkeypatch.setattr(
        "trustrag.modules.reranker.bge_reranker."
        "AutoModelForSequenceClassification.from_pretrained",
        model_loader,
    )

    config = BgeRerankerConfig(model_name_or_path="model", device=device)
    reranker = BgeReranker(config)

    tokenizer_loader.assert_called_once_with("model")
    model_loader.assert_called_once_with("model")
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    assert model.half.called is uses_half_precision
    assert reranker.device == device
