import os
import json
import aiohttp
import httpx
from typing import Dict, List, Any, Union, Optional
from loguru import logger

from .config import get_config


class SerperClient:
    """Client for the Serper.dev API to perform web searches."""
    
    def __init__(self):
        config = get_config()
        self.api_key = os.environ.get("SERPER_API_KEY", config.get("serper", {}).get("api_key", ""))
        
        if not self.api_key:
            logger.warning("No Serper API key found. Searches will fail.")
        
        self.api_url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        self.organic_urls = []
        
    def search_sync(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search using the Serper API.
        
        Args:
            query: Search query
            
        Returns:
            List of search result items
        """
        try:
            payload = json.dumps({
                "q": query
            })
            
            response = httpx.post(self.api_url, headers=self.headers, data=payload)
            response.raise_for_status()
            
            result = response.json()
            self.organic_urls = self._extract_urls(result)
            
            # Format the results for consumption
            formatted_results = self._format_results(result)
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching with Serper: {str(e)}")
            return []
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform an async search using the Serper API.
        
        Args:
            query: Search query
            
        Returns:
            List of search result items
        """
        try:
            payload = json.dumps({
                "q": query
            })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, data=payload) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        logger.error(f"Serper API error: {result}")
                        return []
                    
                    self.organic_urls = self._extract_urls(result)
                    
                    # Format the results for consumption
                    formatted_results = self._format_results(result)
                    return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching with Serper: {str(e)}")
            return []
    
    def _extract_urls(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract URLs from search results.
        
        Args:
            result: Search result object
            
        Returns:
            List of URLs
        """
        urls = []
        
        # Extract organic results
        if "organic" in result:
            for item in result["organic"]:
                if "link" in item:
                    urls.append(item["link"])
        
        return urls
    
    def _format_results(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format search results into a standardized format.
        
        Args:
            result: Search result object
            
        Returns:
            List of formatted result items
        """
        formatted_results = []
        
        # Format organic results
        if "organic" in result:
            for item in result["organic"]:
                formatted_item = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "content": f"{item.get('title', '')} - {item.get('snippet', '')}"
                }
                formatted_results.append(formatted_item)
        
        # Include featured snippet if available
        if "answerBox" in result:
            answer_box = result["answerBox"]
            formatted_item = {
                "title": answer_box.get("title", "Featured Snippet"),
                "url": answer_box.get("link", ""),
                "snippet": answer_box.get("snippet", ""),
                "content": answer_box.get("answer", answer_box.get("snippet", ""))
            }
            formatted_results.insert(0, formatted_item)
        
        return formatted_results
    
    def get_organic_urls(self) -> List[str]:
        """
        Get the URLs of organic search results from the last search.
        
        Returns:
            List of URLs
        """
        return self.organic_urls
