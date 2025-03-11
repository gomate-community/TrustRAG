import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import os

import loguru

from trustrag.modules.engine.websearch import DuckduckEngine, SearxngEngine
from trustrag.modules.engine.qdrant import QdrantEngine
from trustrag.modules.retrieval.embedding import SentenceTransformerEmbedding


@dataclass
class SearchResult:
    """Standardized search result format regardless of the search engine used."""

    title: str
    url: str
    description: str
    position: int
    metadata: Dict[str, Any] = None


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @abstractmethod
    async def search_async(
            self, query: str, top_k: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search and return standardized results."""
        pass


class UnifiedSearchEngine(SearchEngine):
    """
    A unified search engine that can use either DuckduckEngine, SearxngEngine,
    or QdrantEngine based on the engine_type parameter.
    """

    def __init__(
            self,
            engine_type: str = "searxng",
            proxy: Optional[str] = None,
            timeout: int = 20,
            searxng_url: str = os.getenv("SEARXNG_URL", "http://localhost:8080/search"),
            qdrant_collection_name: str = "default_collection",
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            embedding_model_path: str = None,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the UnifiedSearchEngine class.

        :param engine_type: Type of search engine to use, either "duckduckgo", "searxng", or "qdrant"
        :param proxy: Proxy address for DuckDuckGo
        :param timeout: Request timeout in seconds
        :param searxng_url: URL of the SearxNG instance if using searxng
        :param qdrant_collection_name: Collection name for Qdrant
        :param qdrant_host: Qdrant server host
        :param qdrant_port: Qdrant server port
        :param embedding_model_path: Path to the embedding model for Qdrant
        :param device: Device to use for embedding model ("cpu" or "cuda")
        """
        self.engine_type = engine_type.lower()

        if self.engine_type == "duckduckgo":
            self.engine = DuckduckEngine(proxy=proxy, timeout=timeout)
        elif self.engine_type == "searxng":
            loguru.logger.info(searxng_url)
            self.engine = SearxngEngine(searxng_url=searxng_url, timeout=timeout)
        elif self.engine_type == "qdrant":
            loguru.logger.info("Using Arxiv Qdrant")
            if not embedding_model_path:
                raise ValueError("embedding_model_path must be provided for Qdrant engine")

            # Initialize the embedding model
            self.embedding_generator = SentenceTransformerEmbedding(
                model_name_or_path=embedding_model_path,
                device=device
            )

            # Initialize Qdrant engine
            self.engine = QdrantEngine(
                collection_name=qdrant_collection_name,
                embedding_generator=self.embedding_generator,
                qdrant_client_params={"host": qdrant_host, "port": qdrant_port},
                vector_size=self.embedding_generator.embedding_size
            )
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}. Choose 'duckduckgo', 'searxng', or 'qdrant'")

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform a search and return standardized results.

        :param query: Search keyword(s)
        :param top_k: Maximum number of results to return
        :param kwargs: Additional search parameters for the specific engine
        :return: List of standardized search results
        """
        if self.engine_type == "duckduckgo":
            # Use DuckDuckGo engine
            raw_results = self.engine.search(query, top_k)
        elif self.engine_type == "searxng":
            # Use SearxNG engine
            raw_results = self.engine.search(query, top_k, **kwargs)
        elif self.engine_type == "qdrant":
            # Use Qdrant engine
            raw_results = self.engine.search(text=query, limit=top_k, **kwargs)
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")

        # Convert to standardized format
        standardized_results = []
        for i, result in enumerate(raw_results, start=1):
            # Handle different result formats from different engines
            if self.engine_type == "duckduckgo":
                title = result.get("title", "")
                url = result.get("href", "")
                description = result.get("body", "")
                metadata = result
            elif self.engine_type == "searxng":
                title = result.get("title", "")
                url = result.get("url", "")
                description = result.get("content", "")
                metadata = result
            else:  # qdrant
                payload = result.get("payload", {})
                score = result.get("score", 0.0)

                title = payload.get("title", "")
                url = payload.get("url", "") if "url" in payload else f"qdrant://{i}"
                description = payload.get("abstract", "") or payload.get("summary", "")

                # Include score in metadata
                metadata = {
                    "original_payload": payload,
                    "score": score
                }

            standardized_results.append(
                SearchResult(
                    title=title,
                    url=url,
                    description=description,
                    position=i,
                    metadata=metadata
                )
            )

        return standardized_results

    async def search_async(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform an asynchronous search and return standardized results.

        :param query: Search keyword(s)
        :param top_k: Maximum number of results to return
        :param kwargs: Additional search parameters for the specific engine
        :return: List of standardized search results
        """
        # Use run_in_executor to make synchronous search method async
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: self.search(query, top_k, **kwargs)
        )
        return results

    def print_results(self, results: List[SearchResult]) -> None:
        """
        Print the search results in a readable format.

        :param results: List of standardized search results
        """
        for result in results:
            print(f"Result {result.position}:")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Description: {result.description}")
            if self.engine_type == "qdrant":
                print(f"Score: {result.metadata.get('score', 'N/A')}")
            print()

if __name__ == '__main__':
    # 使用DuckDuckGo引擎
    # duck_search = DeepSearchEngine(engine_type="duckduckgo")
    # results = duck_search.search("机器学习教程", top_k=3)
    # duck_search.print_results(results)

    # # 使用SearxNG引擎
    # searx_search = UnifiedSearchEngine(
    #     engine_type="searxng",
    #     searxng_url="http://localhost:8080/search"
    # )
    # results = searx_search.search(
    #     "大模型强化学习技术",
    #     top_k=3,
    #     language="zh-CN",
    #     categories="general"
    # )
    # print(results)
    # searx_search.print_results(results)

    # async def search_example():
    #     engine = DeepSearchEngine()
    #     results = await engine.search_async("async Python", top_k=5)
    #     engine.print_results(results)
    #

    # 使用Qdrant引擎
    qdrant_search = UnifiedSearchEngine(
        engine_type="qdrant",
        qdrant_collection_name="arxiv_llms",
        qdrant_host="192.168.1.5",
        qdrant_port=6333,
        embedding_model_path="G:/pretrained_models/mteb/bge-m3",
        device="cuda"
    )
    results = qdrant_search.search("Chain of thought", top_k=5)
    qdrant_search.print_results(results)
