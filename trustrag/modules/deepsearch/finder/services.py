from enum import Enum
from typing import Dict, Optional, Any, List, TypedDict
import os
import asyncio
# from trustrag.modules.deepsearch.utils import logger
import loguru
from firecrawl import FirecrawlApp
from trustrag.modules.deepsearch.finder.manager import SearchAndScrapeManager


class SearchServiceType(Enum):
    """Supported search service types."""

    FIRECRAWL = "firecrawl"
    PLAYWRIGHT_DDGS = "playwright_ddgs"


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class SearchService:
    """Unified search service that supports multiple implementations."""

    def __init__(self, service_type: Optional[str] = None):
        """Initialize the appropriate search service.

        Args:
            service_type: The type of search service to use. Defaults to env var or playwright_ddgs.
        """
        # Determine which service to use
        if service_type is None:
            service_type = os.environ.get("DEFAULT_SCRAPER", "playwright_ddgs")

        self.service_type = service_type

        # Initialize the appropriate service
        if service_type == SearchServiceType.FIRECRAWL.value:
            self.firecrawl = Firecrawl(
                api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
                api_url=os.environ.get("FIRECRAWL_BASE_URL"),
            )
            self.manager = None
        else:
            self.firecrawl = None
            self.manager = SearchAndScrapeManager()
            # Initialize resources asynchronously later
            self._initialized = False

    async def ensure_initialized(self):
        """Ensure the service is initialized."""
        if self.manager and not getattr(self, "_initialized", False):
            await self.manager.setup()
            self._initialized = True

    async def cleanup(self):
        """Clean up resources."""
        if self.manager and getattr(self, "_initialized", False):
            await self.manager.teardown()
            self._initialized = False

    async def search(self, query: str, limit: int = 5, **kwargs) -> Dict[str, Any]:
        """Search using the configured service.

        Returns data in a format compatible with the Firecrawl response format.
        """
        await self.ensure_initialized()

        try:
            if self.service_type == SearchServiceType.FIRECRAWL.value:
                return await self.firecrawl.search(query, limit=limit, **kwargs)
            else:
                search_results = await self.manager.search(
                    query, num_results=limit, **kwargs
                )

                scraped_data = await self.manager.search_and_scrape(
                    query, num_results=limit, scrape_all=True, **kwargs
                )

                # Format the response to match Firecrawl format
                formatted_data = []
                for result in search_results:
                    item = {
                        "url": result.url,
                        "title": result.title,
                        "content": "",  # Default empty content
                    }

                    # Add content if we scraped it
                    if result.url in scraped_data["scraped_contents"]:
                        scraped = scraped_data["scraped_contents"][result.url]
                        item["content"] = scraped.text

                    formatted_data.append(item)

                return {"data": formatted_data}

        except Exception as e:
            loguru.logger.error(f"Error during search: {str(e)}")
            return {"data": []}


class Firecrawl:
    """Simple wrapper for Firecrawl SDK."""

    def __init__(self, api_key: str = "", api_url: Optional[str] = None):
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)

    async def search(
        self, query: str, timeout: int = 15000, limit: int = 5
    ) -> SearchResponse:
        """Search using Firecrawl SDK in a thread pool to keep it async."""
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.app.search(
                    query=query,
                ),
            )

            # Handle the response format from the SDK
            if isinstance(response, dict) and "data" in response:
                # Response is already in the right format
                return response
            elif isinstance(response, dict) and "success" in response:
                # Response is in the documented format
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                # Response is a list of results
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        # Handle non-dict items (like objects)
                        formatted_data.append(
                            {
                                "url": getattr(item, "url", ""),
                                "content": getattr(item, "markdown", "")
                                or getattr(item, "content", ""),
                                "title": getattr(item, "title", "")
                                or getattr(item, "metadata", {}).get("title", ""),
                            }
                        )
                return {"data": formatted_data}
            else:
                print(f"Unexpected response format from Firecrawl: {type(response)}")
                return {"data": []}

        except Exception as e:
            print(f"Error searching with Firecrawl: {e}")
            print(
                f"Response type: {type(response) if 'response' in locals() else 'N/A'}"
            )
            return {"data": []}


# Initialize a global instance with the default settings
search_service = SearchService(
    service_type=os.getenv("DEFAULT_SCRAPER", "playwright_ddgs")
)