import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import os
from trustrag.modules.engine.websearch import DuckduckEngine, SearxngEngine


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
    async def search(
            self, query: str, num_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search and return standardized results."""
        pass


class DeepSearchEngine:
    """
    A unified search engine that can use either DuckduckEngine or SearxngEngine
    based on the engine_type parameter.
    """

    def __init__(
            self,
            engine_type: str = "searxng",
            proxy: Optional[str] = None,
            timeout: int = 20,
            searxng_url: str = os.getenv("SEARXNG_URL")
    ) -> None:
        """
        Initialize the UnifiedSearchEngine class.

        :param engine_type: Type of search engine to use, either "duckduckgo" or "searxng"
        :param proxy: Proxy address for DuckDuckGo
        :param timeout: Request timeout in seconds
        :param searxng_url: URL of the SearxNG instance if using searxng
        """
        self.engine_type = engine_type.lower()

        if self.engine_type == "duckduckgo":
            self.engine = DuckduckEngine(proxy=proxy, timeout=timeout)
        elif self.engine_type == "searxng":
            self.engine = SearxngEngine(searxng_url=searxng_url, timeout=timeout)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}. Choose 'duckduckgo' or 'searxng'")

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform a search and return standardized results.

        :param query: Search keyword(s)
        :param top_k: Maximum number of results to return
        :param kwargs: Additional search parameters for SearxNG
        :return: List of standardized search results
        """
        if self.engine_type == "duckduckgo":
            # Use DuckDuckGo engine
            raw_results = self.engine.search(query, top_k)
        else:
            # Use SearxNG engine
            raw_results = self.engine.search(query, top_k, **kwargs)

        # Convert to standardized format
        standardized_results = []
        for i, result in enumerate(raw_results, start=1):
            # Handle different result formats from different engines
            if self.engine_type == "duckduckgo":
                title = result.get("title", "")
                url = result.get("href", "")
                description = result.get("body", "")
            else:  # searxng
                title = result.get("title", "")
                url = result.get("url", "")
                description = result.get("content", "")

            standardized_results.append(
                SearchResult(
                    title=title,
                    url=url,
                    description=description,
                    position=i,
                    metadata=result
                )
            )

        return standardized_results

    async def search_async(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform an asynchronous search and return standardized results.

        :param query: Search keyword(s)
        :param top_k: Maximum number of results to return
        :param kwargs: Additional search parameters for SearxNG
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
            print(f"Description: {result.description}\n")


if __name__ == '__main__':
    # 使用DuckDuckGo引擎
    # duck_search = DeepSearchEngine(engine_type="duckduckgo")
    # results = duck_search.search("机器学习教程", top_k=3)
    # duck_search.print_results(results)

    # 使用SearxNG引擎
    searx_search = DeepSearchEngine(
        engine_type="searxng",
        searxng_url="http://localhost:8080/search"
    )
    results = searx_search.search(
        "机器学习教程",
        top_k=3,
        language="en-US",
        categories="general"
    )
    print(results)
    # searx_search.print_results(results)

    # async def search_example():
    #     engine = DeepSearchEngine()
    #     results = await engine.search_async("async Python", top_k=5)
    #     engine.print_results(results)
    #
    # # 运行异步示例
    # import asyncio
    # asyncio.run(search_example())
