import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any
from trustrag.modules.deepsearch.utils import logger
from abc import ABC, abstractmethod
from duckduckgo_search import DDGS


# ---- Data Models ----


@dataclass
class SearchResult:
    """Standardized search result format regardless of the search engine used."""

    title: str
    url: str
    description: str
    position: int
    metadata: Dict[str, Any] = None


# ---- Search Engine Interfaces ----


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @abstractmethod
    async def search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search and return standardized results."""
        pass


class DdgsSearchEngine:
    """DuckDuckGo search engine implementation."""

    def __init__(self):
        self.ddgs = DDGS()

    async def search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search using DDGS and return standardized results."""
        try:
            # Convert to async operation
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: list(self.ddgs.text(query, max_results=num_results))
            )

            # Convert to standardized format
            standardized_results = []
            for i, result in enumerate(results):
                standardized_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", ""),
                        description=result.get("body", ""),
                        position=i + 1,
                        metadata=result,
                    )
                )

            return standardized_results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []