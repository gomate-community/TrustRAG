from trustrag.modules.retrieval.web_retriever import DuckduckSearcher


if __name__ == "__main__":
    # Use a proxy
    # searcher = DuckduckSearcher(proxy="socks5h://user:password@geo.iproyal.com:32325", timeout=20)
    searcher = DuckduckSearcher(proxy=None, timeout=20)
    # Search for "python programming"
    results = searcher.retrieve("韩国总统 尹锡悦", top_k=5)
    # Print the search results
    searcher.print_results(results)