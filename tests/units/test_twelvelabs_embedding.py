# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from trustrag.modules.vector.embedding import (
    EmbeddingFactory,
    TwelveLabsEmbedding,
    TwelveLabsVideoAnalyzer,
)

requires_key = pytest.mark.skipif(
    not os.getenv("TWELVELABS_API_KEY"),
    reason="TWELVELABS_API_KEY not set",
)


def test_twelvelabs_registered_in_factory():
    """No-network: the provider is wired into the factory."""
    assert "twelvelabs" in EmbeddingFactory.get_available_embedding_types()


def test_video_analyzer_requires_a_source():
    """No-network: analyze() rejects a call with no video source."""
    analyzer = TwelveLabsVideoAnalyzer(api_key="dummy")
    with pytest.raises(ValueError):
        analyzer.analyze(prompt="Summarize this video.")


@requires_key
def test_marengo_text_embedding_is_512_dim():
    generator = TwelveLabsEmbedding()
    embeddings = generator.generate_embeddings(["a cat playing the piano"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 512)


if __name__ == "__main__":
    test_twelvelabs_registered_in_factory()
    test_video_analyzer_requires_a_source()
    if os.getenv("TWELVELABS_API_KEY"):
        test_marengo_text_embedding_is_512_dim()
    print("ok")
