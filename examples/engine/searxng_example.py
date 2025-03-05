from trustrag.modules.engine.websearch import SearxngEngine, DuckduckEngine

if __name__ == "__main__":
    search_engine = SearxngEngine()
    # Create an instance of SearxngEngine
    searxng = SearxngEngine()

    # Perform a search
    search_results = searxng.search(
        "医疗的梅奥 默沙东",
        language="zh-CN",
        top_k=50
    )

    # Print results
    for result in search_results:
        print(result)