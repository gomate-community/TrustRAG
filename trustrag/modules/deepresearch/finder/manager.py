import asyncio
from typing import List, Dict, Union
# from trustrag.modules.deepsearch.utils import logger
import loguru
from trustrag.modules.deepresearch.finder.searcher import SearchResult, SearchEngine, UnifiedSearchEngine
from trustrag.modules.deepresearch.finder.scraper import ScrapedContent, Scraper, PlaywrightScraper


class SearchAndScrapeManager:
    """Main class for coordinating search and scrape operations."""

    def __init__(self, search_engine: SearchEngine = None, scraper: Scraper = None):
        self.search_engine = search_engine or UnifiedSearchEngine()
        self.scraper = scraper or PlaywrightScraper()

    async def setup(self):
        """Initialize required resources."""
        if hasattr(self.scraper, "setup"):
            await self.scraper.setup()

    async def teardown(self):
        """Clean up resources."""
        if hasattr(self.scraper, "teardown"):
            await self.scraper.teardown()

    async def search(
        self, query: str, num_results: int = 5, **kwargs
    ) -> List[SearchResult]:
        """Perform a search using the configured search engine."""
        return await self.search_engine.search_async(query, num_results, **kwargs)

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Scrape a URL using the configured scraper."""
        return await self.scraper.scrape(url, **kwargs)

    async def search_and_scrape(
        self,
        query: str,
        num_results: int = 5,
        scrape_all: bool = False,
        max_concurrent_scrapes: int = 5,
        **kwargs,
    ) -> Dict[str, Union[List[SearchResult], Dict[str, ScrapedContent]]]:
        """
        Search for results and optionally scrape them.

        Args:
            query: Search query string
            num_results: Maximum number of search results to retrieve
            scrape_all: Whether to scrape all search results
            max_concurrent_scrapes: Maximum number of concurrent scrape operations
            **kwargs: Additional parameters to pass to search and scrape methods

        Returns:
            Dictionary containing search results and scraped content
        """
        # Perform search
        search_results = await self.search(query, num_results, **kwargs)

        scraped_contents = {}

        # Scrape results if requested
        if scrape_all and search_results:
            # Create a semaphore to limit concurrent scrapes
            semaphore = asyncio.Semaphore(max_concurrent_scrapes)

            async def scrape_with_semaphore(url):
                loguru.logger.info("Scraping url==>:"+url)
                async with semaphore:
                    return await self.scrape(url, **kwargs)

            # Create scraping tasks
            scrape_tasks = [
                scrape_with_semaphore(result.url) for result in search_results
            ]

            # Execute scraping tasks concurrently with rate limiting
            scraped_results = await asyncio.gather(
                *scrape_tasks, return_exceptions=True
            )

            # Process results
            for i, result in enumerate(scraped_results):
                if isinstance(result, Exception):
                    loguru.logger.error(f"Error scraping result {i+1}: {str(result)}")
                    continue

                scraped_contents[search_results[i].url] = result

        return {"search_results": search_results, "scraped_contents": scraped_contents}