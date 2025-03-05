from typing import List, Dict, Optional
from duckduckgo_search import DDGS

from typing import List, Dict, Optional
import requests


class DuckduckEngine:
    def __init__(self, proxy: Optional[str] = None, timeout: int = 20) -> None:
        """
        Initialize the DuckduckSearcher class.

        :param proxy: Proxy address, e.g., "socks5h://user:password@geo.iproyal.com:32325"
        :param timeout: Request timeout in seconds, default is 20
        """
        self.proxy = proxy
        self.timeout = timeout
        self.ddgs = DDGS(proxy=self.proxy, timeout=self.timeout)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Perform a search and return the results.

        :param query: Search keyword(s)
        :param top_k: Maximum number of results to return, default is 5
        :return: List of search results, each result is a dictionary with keys 'title', 'href', and 'body'
        """
        results = self.ddgs.text(query, max_results=top_k)
        return results

    def print_results(self, results: List[Dict[str, str]]) -> None:
        """
        Print the search results in a readable format.

        :param results: List of search results, each result is a dictionary with keys 'title', 'href', and 'body'
        """
        for i, result in enumerate(results, start=1):
            print(f"Result {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Body: {result['body']}\n")



class SearxngEngine:
    def __init__(
        self,
        searxng_url: str = "http://localhost:8080/search",
        timeout: int = 20
    ) -> None:
        """
        Initialize the SearxngEngine class.

        :param searxng_url: URL of the SearxNG instance, default is "http://localhost:8080/search"
        :param timeout: Request timeout in seconds, default is 20
        """
        self.searxng_url = searxng_url
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36"
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Perform a search and return the results.

        :param query: Search keyword(s)
        :param top_k: Maximum number of results to return, default is 5
        :param kwargs: Additional search parameters (e.g., language, categories, etc.)
        :return: List of search results
        """
        # Prepare parameters
        params = {
            "q": query,
            "format": kwargs.get("output_format", "json"),
            "page": kwargs.get("page", 1),
        }

        # Optional parameters
        optional_params = {
            "categories": kwargs.get("categories"),
            "engines": kwargs.get("engines"),
            "language": kwargs.get("language"),
            "time_range": kwargs.get("time_range"),
            "results_on_new_tab": kwargs.get("results_on_new_tab"),
            "image_proxy": kwargs.get("image_proxy"),
            "autocomplete": kwargs.get("autocomplete"),
            "safesearch": kwargs.get("safesearch"),
            "theme": kwargs.get("theme"),
            "enabled_plugins": kwargs.get("enabled_plugins"),
            "disabled_plugins": kwargs.get("disabled_plugins"),
            "enabled_engines": kwargs.get("enabled_engines"),
            "disabled_engines": kwargs.get("disabled_engines"),
        }

        # Add non-None optional parameters
        params.update({k: v for k, v in optional_params.items() if v is not None})

        try:
            # Perform the search request
            response = requests.get(
                self.searxng_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )

            # Check response status
            if response.status_code != 200:
                raise Exception(f"Search query failed with status code {response.status_code}")

            # Extract and return results, limiting to top_k
            results = response.json().get("results", [])
            return results[:top_k]

        except requests.RequestException as e:
            print(f"Request error: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Create an instance of SearxngEngine
    searxng = SearxngEngine()

    # Perform a search
    search_results = searxng.search(
        "医疗的梅奥 默沙东",
        language="zh-CN",
        top_k=5
    )

    # Print results
    for result in search_results:
        print(result)

    # Use a proxy
    # searcher = DuckduckSearcher(proxy="socks5h://user:password@geo.iproyal.com:32325", timeout=20)
    searcher = DuckduckEngine(proxy=None, timeout=20)
    # Search for "python programming"
    results = searcher.search("怎么学习机器学习", top_k=5)
    print(results)
    # Print the search results
    searcher.print_results(results)