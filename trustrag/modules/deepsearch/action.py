import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict, TypedDict, Optional

import loguru
import openai

from trustrag.modules.deepsearch.agent.providers import trim_prompt, get_client_response
from trustrag.modules.deepsearch.finder.services import search_service
from trustrag.modules.prompt.templates import DEEPSEARCH_SYSTEM_PROMPT


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]


@dataclass
class SerpQuery:
    query: str
    research_goal: str


async def generate_serp_queries(
        query: str,
        client: openai.OpenAI,
        model: str,
        num_queries: int = 3,
        learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Generate SERP queries based on user input and previous learnings."""
    print("正在进行研究主题的意图理解...")

    prompt = f"""根据用户的以下提示，生成一系列SERP查询来研究该主题。
    返回一个JSON对象，其中包含一个'queries'数组字段，包含{num_queries}个查询（如果原始提示已经很明确，则可以少于这个数量）。
    每个查询对象应该有'query'和'research_goal'字段。
    请注意JSON对象一定要格式正确，不要输出其他额外内容。
    确保每个查询都是唯一的，且彼此不相似：主题<prompt>{query}</prompt>"""
    if learnings:
        # prompt += f"\n\nHere are some learnings from previous research, use them to generate more specific queries: {' '.join(learnings)}"
        prompt += f"\n\n这里是之前研究步骤的一些发现，请使用它们生成更具体的查询：{' '.join(learnings)}。请确保生成的查询与用户原始提示的语言保持一致。"
        # prompt += f"\n\n以下是从以前的研究步骤中得到的一些经验，请使用它们来生成更具体的查询：{' '.join(learnings)}"
    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": DEEPSEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        queries = response.get("queries", [])
        return [SerpQuery(**q) for q in queries][:num_queries]
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return []


async def process_serp_result(
        query: str,
        search_result: SearchResponse,
        client: openai.OpenAI,
        model: str,
        num_learnings: int = 3,
        num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    contents = [
        trim_prompt(item.get("content", ""), 25_000)
        for item in search_result["data"]
        if item.get("content")
    ]

    # Create the contents string separately
    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    # prompt = (
    #     f"根据以下对查询<query>{query}</query>的网页搜索内容，"
    #     f"生成从内容中得到的学习要点列表。返回一个JSON对象，包含'learnings'和'followUpQuestions'键(key)，"
    #     f"值(value)为字符串数组。包括最多{num_learnings}个学习要点和{num_follow_up_questions}个后续问题。"
    #     f"学习要点应该独特、简洁且信息丰富，包括实体、指标、数字和日期。\n\n"
    #     f"<contents>{contents_str}</contents>"
    #     f"请确保生成的学习要点和后续问题与用户原始查询({query})的语言保持一致。"
    # )

    prompt = (
        f"基于以下来自搜索引擎结果页面(SERP)对查询<query>{query}</query>的内容，"
        f"生成一个内容要点列表。返回一个JSON对象，包含'learnings'和'followUpQuestions'两个键，"
        f"learnings'代表从内容中学习获取的关键知识要点列表[xx,xxx]，应该是独特、简洁且信息丰富的，可以包含概念名称、指标、数字和日期等。"
        f"'followUpQuestions'代表根据内容生成的推荐问题列表[问题1,问题2,xxx]，用于引导更深入的探索。"
        f"JSON对象格式一定要正确，不要输出额外内容，不要漏掉learnings和followUpQuestions键值对"
        f"learnings和followUpQuestions的值为字符串列表。包括最多{num_learnings}个要点和{num_follow_up_questions}个后续问题。"
        f"<contents>{contents_str}</contents>"
    )

    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": DEEPSEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    loguru.logger.info(response)
    try:
        return {
            "learnings": response.get("learnings", [])[:num_learnings],
            "followUpQuestions": response.get("followUpQuestions", [])[
                                 :num_follow_up_questions
                                 ],
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return {"learnings": [], "followUpQuestions": []}


async def write_final_report(
        prompt: str,
        learnings: List[str],
        visited_urls: List[str],
        client: openai.OpenAI,
        model: str,
) -> str:
    """Generate final report based on all research learnings."""

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        # 150_000,
        300_000,
    )

    user_prompt = (
        f"根据以下用户提供的提示，使用研究中获得的学习要点撰写关于该主题的最终报告。返回一个JSON对象，"
        f"其中包含'reportMarkdown'字段，该字段包含详细的markdown报告（目标为3页以上），尽量内容丰富饱满。包括研究中的所有学习要点：\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"以下是研究中获得的所有学习要点：\n\n<learnings>\n{learnings_string}\n</learnings>"
    )
    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": DEEPSEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        report = response.get("reportMarkdown", "")

        # Append sources
        urls_section = "\n\n## 来源\n\n" + "\n".join(
            [f"- {url}" for url in visited_urls]
        )
        return report + urls_section
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return "Error generating report"


async def deep_research(
        query: str,
        breadth: int,
        depth: int,
        concurrency: int,
        client: openai.OpenAI,
        model: str,
        learnings: List[str] = None,
        visited_urls: List[str] = None,
) -> ResearchResult:
    """
    Main research function that recursively explores a topic.

    Args:
        query: Research query/topic
        breadth: Number of parallel searches to perform
        depth: How many levels deep to research
        concurrency: Number of concurrent requests allowed
        client: OpenAI client instance
        model: Model name to use for API calls
        learnings: Previous learnings to build upon
        visited_urls: Previously visited URLs

    Returns:
        ResearchResult: Dictionary containing learnings and visited URLs
    """
    learnings = learnings or []
    visited_urls = visited_urls or []

    loguru.logger.info(f"开始研究查询: '{query}'")
    loguru.logger.info(f"当前深度: {depth}, 广度: {breadth}, 并发数: {concurrency}")

    # Generate search queries
    serp_queries = await generate_serp_queries(
        query=query,
        client=client,
        model=model,
        num_queries=breadth,
        learnings=learnings,
    )

    # 打印所有生成的查询
    loguru.logger.info(f"生成了 {len(serp_queries)} 个搜索查询:")
    for i, serp_query in enumerate(serp_queries):
        loguru.logger.info(f"  查询 {i + 1}: {serp_query.query}")
        loguru.logger.debug(f"  研究目标: {serp_query.research_goal}")

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)

    async def process_query(serp_query: SerpQuery) -> ResearchResult:
        async with semaphore:
            try:
                # Search for content
                query_id = serp_query.query[:30] + "..." if len(serp_query.query) > 30 else serp_query.query
                loguru.logger.info(f"开始处理子查询: '{query_id}'")

                result = await search_service.search(serp_query.query, limit=2)
                url_count = len([item for item in result["data"] if item.get("url")])
                loguru.logger.info(f"子查询 '{query_id}' 返回了 {url_count} 个URL")

                # Collect new URLs
                new_urls = [
                    item.get("url") for item in result["data"] if item.get("url")
                ]

                # Log the URLs found
                for url in new_urls:
                    loguru.logger.debug(f"发现URL: {url}")

                # Calculate new breadth and depth for next iteration
                new_breadth = max(1, breadth // 2)
                new_depth = depth - 1

                # Process the search results
                loguru.logger.info(f"处理子查询 '{query_id}' 的搜索结果")
                new_learnings = await process_serp_result(
                    query=serp_query.query,
                    search_result=result,
                    num_follow_up_questions=new_breadth,
                    client=client,
                    model=model,
                )

                loguru.logger.info(f"子查询 '{query_id}' 获得了 {len(new_learnings['learnings'])} 条学习内容")
                all_learnings = learnings + new_learnings["learnings"]
                all_urls = visited_urls + new_urls

                # If we have more depth to go, continue research
                if new_depth > 0:
                    loguru.logger.info(
                        f"继续深入研究子查询 '{query_id}', 新广度: {new_breadth}, 新深度: {new_depth}"
                    )

                    next_query = f"""
                    Previous research goal: {serp_query.research_goal}
                    Follow-up research directions: {" ".join(new_learnings["followUpQuestions"])}
                    """.strip()

                    return await deep_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        concurrency=concurrency,
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        client=client,
                        model=model,
                    )

                loguru.logger.info(f"子查询 '{query_id}' 研究完成")
                return {"learnings": all_learnings, "visited_urls": all_urls}

            except Exception as e:
                if "Timeout" in str(e):
                    loguru.logger.error(f"查询超时: '{serp_query.query}': {e}")
                else:
                    loguru.logger.error(f"查询出错: '{serp_query.query}': {e}")
                return {"learnings": [], "visited_urls": []}

    # Process all queries concurrently
    loguru.logger.info(f"并发处理 {len(serp_queries)} 个查询")
    results = await asyncio.gather(*[process_query(query) for query in serp_queries])

    # Combine all results
    all_learnings = list(
        set(learning for result in results for learning in result["learnings"])
    )

    all_urls = list(set(url for result in results for url in result["visited_urls"]))

    # 最终打印总结
    loguru.logger.info("=" * 50)
    loguru.logger.info("研究完成")
    loguru.logger.info(f"原始查询: '{query}'")
    loguru.logger.info(f"衍生的搜索查询:")
    for i, serp_query in enumerate(serp_queries):
        loguru.logger.info(f"  {i + 1}. {serp_query.query}")
    loguru.logger.info(f"总共获得 {len(all_learnings)} 条学习内容和 {len(all_urls)} 个URL")
    loguru.logger.info("=" * 50)

    return {"learnings": all_learnings, "visited_urls": all_urls}