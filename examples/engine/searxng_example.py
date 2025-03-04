import requests
from typing import List, Optional


def search_web(
        query: str,
        searxng_url: str = "http://localhost:8080/search",
        categories: Optional[str] = None,
        engines: Optional[str] = None,
        language: Optional[str] = None,
        page: int = 1,
        time_range: Optional[str] = None,
        output_format: str = "json",
        results_on_new_tab: int = 0,
        image_proxy: Optional[bool] = None,
        autocomplete: Optional[str] = None,
        safesearch: Optional[int] = None,
        theme: Optional[str] = "simple",
        enabled_plugins: Optional[str] = None,
        disabled_plugins: Optional[str] = None,
        enabled_engines: Optional[str] = None,
        disabled_engines: Optional[str] = None,
) -> List[dict]:
    """
    Perform a web search using SearXNG API with various optional parameters.

    :param query: Required search query.
    :param searxng_url: URL of the SearXNG instance.
    :param categories: Comma-separated list of search categories.
    :param engines: Comma-separated list of search engines.
    :param language: Language code for search results.
    :param page: Search results page number.
    :param time_range: Time filter (day, month, year) for engines supporting it.
    :param output_format: Format of results (json, csv, rss).
    :param results_on_new_tab: Whether to open results in a new tab (0 or 1).
    :param image_proxy: Proxy image results through SearXNG.
    :param autocomplete: Autocomplete service.
    :param safesearch: Safe search filtering level (0, 1, 2).
    :param theme: Theme of the instance.
    :param enabled_plugins: Comma-separated list of enabled plugins.
    :param disabled_plugins: Comma-separated list of disabled plugins.
    :param enabled_engines: Comma-separated list of enabled engines.
    :param disabled_engines: Comma-separated list of disabled engines.
    :return: List of search results.
    """
    params = {
        "q": query,
        "format": output_format,
        "page": page,
    }

    # Add optional parameters if provided
    optional_params = {
        "categories": categories,
        "engines": engines,
        "language": language,
        "time_range": time_range,
        "results_on_new_tab": results_on_new_tab,
        "image_proxy": image_proxy,
        "autocomplete": autocomplete,
        "safesearch": safesearch,
        "theme": theme,
        "enabled_plugins": enabled_plugins,
        "disabled_plugins": disabled_plugins,
        "enabled_engines": enabled_engines,
        "disabled_engines": disabled_engines,
    }

    # Only add parameters that are not None
    params.update({k: v for k, v in optional_params.items() if v is not None})

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }

    response = requests.get(searxng_url, params=params, headers=headers)

    if response.status_code != 200:
        print("Response status code:", response.status_code)
        print("Response text:", response.text)
        raise Exception(f"Search query failed with status code {response.status_code}")

    return response.json().get("results", [])


# Example usage:
search_results = search_web("怎么学习机器学习", language="zh-CN", page=1)
print(search_results)

for content in search_results:
    print(content)