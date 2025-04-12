import arxiv
import os
import json
import time
from tqdm import tqdm
import logging
from datetime import datetime

def process_metadata(result):
    """
    将ArXiv结果对象转换为结构化的字典

    Args:
        result: arxiv.Result对象

    Returns:
        dict: 结构化的元数据字典
    """
    metadata = {
        "entry_id": result.entry_id,
        "updated": str(result.updated),
        "published": str(result.published),
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "summary": result.summary,
        "comment": str(result.comment),
        "journal_ref": str(result.journal_ref),
        "doi": str(result.doi),
        "primary_category": result.primary_category,
        "categories": result.categories,
        "links": [{"title": link.title, "href": link.href, "rel": link.rel} for link in result.links],
        "pdf_url": result.pdf_url,
        "download_time": datetime.now().isoformat()
    }

    return metadata


def download_arxiv_papers(topic, max_papers=200, save_dir="papers", sleep_interval=2):
    """
    下载指定主题的ArXiv论文并保存结构化元数据

    Args:
        topic (str): 要搜索的主题/查询
        max_papers (int): 要下载的最大论文数量
        save_dir (str): 保存论文的基本目录
        sleep_interval (float): 下载间隔时间(避免API限制)

    Returns:
        int: 成功下载的论文数量
    """
    # 配置日志
    topic_safe = topic.replace(' ', '_').replace('/', '_').replace('\\', '_')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{save_dir}/{topic_safe}_download.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"开始下载主题: {topic}")

    # 创建文件夹结构
    topic_dir = os.path.join(save_dir, f"topic_{topic_safe}")
    pdfs_dir = os.path.join(topic_dir, "pdfs")
    metadata_dir = os.path.join(topic_dir, "metadata")

    os.makedirs(pdfs_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    logger.info(f"创建目录: {pdfs_dir} 和 {metadata_dir}")

    # 创建一个总体元数据文件，包含所有下载的论文信息
    all_metadata_file = os.path.join(topic_dir, f"{topic_safe}_all_metadata.json")
    all_metadata = []

    # 配置搜索
    search = arxiv.Search(
        query=topic,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client()

    # 初始化计数器
    successful_downloads = 0
    failed_downloads = 0

    # 下载论文
    try:
        results = list(client.results(search))
        total_results = len(results)
        logger.info(f"找到 {total_results} 篇关于主题 '{topic}' 的论文")

        for i, result in enumerate(tqdm(results, desc=f"下载主题 '{topic}' 的论文")):
            try:
                # 获取论文ID并创建文件名
                paper_id = result.get_short_id()
                pdf_filename = f"{paper_id}.pdf"
                metadata_filename = f"{paper_id}.json"

                # 处理元数据
                metadata = process_metadata(result)
                metadata_path = os.path.join(metadata_dir, metadata_filename)

                # 保存单个论文元数据
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                # 添加到总体元数据
                all_metadata.append(metadata)

                # 保存总体元数据每10篇论文更新一次
                if (i + 1) % 10 == 0 or (i + 1) == total_results:
                    with open(all_metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

                # 下载PDF
                pdf_path = os.path.join(pdfs_dir, pdf_filename)
                result.download_pdf(dirpath=pdfs_dir, filename=pdf_filename)
                successful_downloads += 1

                # 休眠以避免速率限制
                time.sleep(sleep_interval)

            except Exception as e:
                logger.error(f"下载论文 {paper_id} 时出错: {str(e)}")
                failed_downloads += 1

            # 每10篇论文记录一次进度
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{total_results} 篇论文已处理")
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"搜索或下载过程中出错: {str(e)}")

    # 记录最终统计信息
    logger.info(f"主题 '{topic}' 的下载已完成")
    logger.info(f"成功下载: {successful_downloads} 篇论文")
    logger.info(f"下载失败: {failed_downloads} 篇论文")

    return successful_downloads


def batch_download_topics(topics_list, max_papers_per_topic=200, base_dir="papers"):
    """
    批量下载多个主题的论文

    Args:
        topics_list (list): 主题列表
        max_papers_per_topic (int): 每个主题要下载的最大论文数量
        base_dir (str): 基本保存目录

    Returns:
        dict: 每个主题的下载统计信息
    """
    os.makedirs(base_dir, exist_ok=True)

    results = {}
    total_start_time = time.time()

    for i, topic in enumerate(topics_list):
        print(f"\n[{i + 1}/{len(topics_list)}] 开始下载主题: {topic}")

        topic_start_time = time.time()
        papers_downloaded = download_arxiv_papers(
            topic=topic,
            max_papers=max_papers_per_topic,
            save_dir=base_dir,
            sleep_interval=3  # 为批量下载增加一点休眠时间
        )

        topic_elapsed_time = time.time() - topic_start_time

        results[topic] = {
            "papers_downloaded": papers_downloaded,
            "elapsed_time": f"{topic_elapsed_time:.2f} 秒"
        }

        print(f"主题 '{topic}' 已完成: 下载 {papers_downloaded} 篇论文，用时 {topic_elapsed_time:.2f} 秒")

        # 在主题之间添加额外休眠以减轻API负担
        if i < len(topics_list) - 1:
            rest_time = 10
            print(f"休息 {rest_time} 秒后继续下一个主题...")
            time.sleep(rest_time)

    total_elapsed_time = time.time() - total_start_time
    print(f"\n批量下载已完成! 总用时: {total_elapsed_time:.2f} 秒")

    # 保存批量下载的摘要
    summary_file = os.path.join(base_dir, "batch_download_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        summary = {
            "total_topics": len(topics_list),
            "total_time": f"{total_elapsed_time:.2f} 秒",
            "completed_at": datetime.now().isoformat(),
            "topics_results": results
        }
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results


# 使用示例:
if __name__ == "__main__":
    # 单个主题下载
    # download_arxiv_papers("Reasoning Large Language Models", max_papers=200)

    # 多个主题批量下载
    topics = [
        # "Reasoning Large Language Models",
        # "LLM Post-Training",
        # "Chain of Thought",
        # "Retrieval Augmented Generation RAG",
        "Continual Learning",
        "Multi-Agent",
        "Tool Learning",
        "Multi-Step Reasoning",
        "Fine-Tuning",
        "Reinforcement Learning Large Language Models LLM",
        "LLM Based Agent",
        "In-Context Learning",
        "RLHF",
        "Pre-Training",
    ]

    batch_download_topics(topics, max_papers_per_topic=1000)