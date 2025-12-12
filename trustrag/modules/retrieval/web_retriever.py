from typing import List, Dict, Optional
from duckduckgo_search import DDGS


class DuckduckSearcher:
    def __init__(self, proxy: Optional[str] = None, timeout: int = 20) -> None:
        """
        Initialize the DuckduckSearcher class.

        :param proxy: Proxy address, e.g., "socks5h://user:password@geo.iproyal.com:32325"
        :param timeout: Request timeout in seconds, default is 20
        """
        self.proxy = proxy
        self.timeout = timeout
        self.ddgs = DDGS(proxy=self.proxy, timeout=self.timeout)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
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


# Example usage
if __name__ == "__main__":
    # Use a proxy
    # searcher = DuckduckSearcher(proxy="socks5h://user:password@geo.iproyal.com:32325", timeout=20)
    searcher = DuckduckSearcher(proxy=None, timeout=20)
    # Search for "python programming"
    results = searcher.retrieve("韩国总统 尹锡悦", top_k=5)
    print(results)
    # Print the search results
    searcher.print_results(results)