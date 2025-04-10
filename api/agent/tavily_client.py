"""
Client for Tavily Search API.

curl -X POST https://api.tavily.com/search \\
-H 'Content-Type: application/json' \\
-H 'Authorization: Bearer tvly-dev-xxx' \\
-d '{
    "query": "good"
}'

Example response:
{
  "query": "good",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "title": "GOOD Definition & Meaning - Merriam-Webster",
      "url": "https://www.merriam-webster.com/dictionary/good",
      "content": "good a good many of us good : something that is good...",
      "score": 0.7283819,
      "raw_content": null
    },
    ...
  ],
  "response_time": 1.38
}
"""
import httpx
import aiohttp
from typing import Dict, Any, Optional
from loguru import logger

from .config import get_config


class TavilyClient:
    """Client for Tavily Search API."""

    def __init__(self):
        config = get_config()
        self.api_key = config.get("tavily", {}).get("api_key")
        self.base_url = config.get("tavily", {}).get("base_url", "https://api.tavily.com/search")
        self.client = httpx.Client(timeout=30.0)

    def search_sync(self, query: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a search using Tavily API.

        Args:
            query: Search query
            options: Additional options for the search

        Returns:
            Dict containing search results
        """
        if not self.api_key:
            raise ValueError("Tavily API key not configured")

        if options is None:
            options = {}

        # Default payload for Tavily API
        payload = {
            "query": query
        }

        # Add optional parameters if provided
        if "search_depth" in options:
            payload["search_depth"] = options.get("search_depth")
        if "include_domains" in options and options["include_domains"]:
            payload["include_domains"] = options.get("include_domains")
        if "exclude_domains" in options and options["exclude_domains"]:
            payload["exclude_domains"] = options.get("exclude_domains")
        if "max_results" in options:
            payload["max_results"] = options.get("max_results")

        # Use Authorization Bearer format as per the official API example
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            logger.debug(f"Searching with Tavily API: {query}")
            response = self.client.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            # Transform the result to match the expected format
            transformed_result = self._transform_result(result, query)
            return transformed_result

        except Exception as e:
            logger.error(f"Error searching with Tavily API: {str(e)}")
            raise

    async def search(self, query: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform an async search using Tavily API.
        
        Args:
            query: Search query
            options: Additional options for the search
            
        Returns:
            Dict containing search results
        """
        if not self.api_key:
            raise ValueError("Tavily API key not configured")

        if options is None:
            options = {}

        # Default payload for Tavily API
        payload = {
            "query": query
        }

        # Add optional parameters if provided
        if "search_depth" in options:
            payload["search_depth"] = options.get("search_depth")
        if "include_domains" in options and options["include_domains"]:
            payload["include_domains"] = options.get("include_domains")
        if "exclude_domains" in options and options["exclude_domains"]:
            payload["exclude_domains"] = options.get("exclude_domains")
        if "max_results" in options:
            payload["max_results"] = options.get("max_results")

        # Use Authorization Bearer format as per the official API example
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            logger.debug(f"Searching with Tavily API: {query}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    # Transform the result to match the expected format
                    transformed_result = self._transform_result(result, query)
                    return transformed_result

        except Exception as e:
            logger.error(f"Error searching with Tavily API: {str(e)}")
            raise

    def _transform_result(self, tavily_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Transform Tavily API result to match the expected format.
        
        Args:
            tavily_result: Raw result from Tavily API
            query: Original search query
            
        Returns:
            Transformed result in compatible format
        """
        transformed_data = []

        # Process results from Tavily API
        if "results" in tavily_result and isinstance(tavily_result["results"], list):
            for item in tavily_result["results"]:
                content = ""

                # Extract content from the result
                if "content" in item and item["content"]:
                    content += item["content"] + "\n\n"

                # Add score information if available
                if "score" in item:
                    content += f"Relevance Score: {item['score']}\n\n"

                transformed_item = {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": content.strip(),
                    "source": "tavily"
                }
                transformed_data.append(transformed_item)

        # If answer is provided by Tavily, add it as a special result
        if "answer" in tavily_result and tavily_result["answer"]:
            transformed_item = {
                "url": "",
                "title": "Tavily Direct Answer",
                "content": tavily_result["answer"],
                "source": "tavily_answer"
            }
            transformed_data.append(transformed_item)

        # If follow-up questions are provided, add them as a special result
        if "follow_up_questions" in tavily_result and tavily_result["follow_up_questions"]:
            follow_up_content = "Suggested follow-up questions:\n\n"
            for question in tavily_result["follow_up_questions"]:
                follow_up_content += f"- {question}\n"

            transformed_item = {
                "url": "",
                "title": "Suggested Follow-up Questions",
                "content": follow_up_content,
                "source": "tavily_follow_up"
            }
            transformed_data.append(transformed_item)

        # If no results were found or processing failed
        if not transformed_data:
            transformed_item = {
                "url": "",
                "title": f"Tavily Search Result for: {query}",
                "content": "No results found or could not process the search response.",
                "source": "tavily"
            }
            transformed_data.append(transformed_item)

        # Add response time if available
        if "response_time" in tavily_result:
            logger.debug(f"Tavily search completed in {tavily_result['response_time']} seconds")

        return {
            "query": query,
            "data": transformed_data
        }