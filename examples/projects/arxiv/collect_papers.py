import os
import json
import pandas as pd
from pathlib import Path

# 定义主题目录
TOPIC_DIRECTORIES = [
    "papers/topic_Chain_of_Thought",
    "papers/topic_LLM_Post-Training",
    "papers/topic_Reasoning_Large_Language_Models",
    "papers/topic_Retrieval_Augmented_Generation_RAG",
    "papers/topic_Continual_Learning",
    "papers/topic_Multi-Agent",
    "papers/topic_Tool_Learning",
    "papers/topic_Multi-Step_Reasoning",
    "papers/topic_Fine-Tuning",
    "topic_LLM_Based_Agent",
    "topic_In-Context_Learning",
    "topic_RLHF",
    "topic_Pre-Training"
]

# 用于存储所有论文数据的列表
all_papers = []

# 遍历每个主题目录
for topic_dir in TOPIC_DIRECTORIES:
    # 构建metadata和output的完整路径
    metadata_dir = os.path.join( topic_dir, "metadata")
    output_dir = os.path.join(topic_dir, "output")

    # 确保目录存在
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        continue

    # 遍历output目录中的所有子目录（每个子目录对应一篇论文）
    for paper_id in os.listdir(output_dir):
        paper_output_dir = os.path.join(output_dir, paper_id)

        # 检查是否是目录
        if not os.path.isdir(paper_output_dir):
            continue

        # 查找markdown文件
        md_files = list(Path(paper_output_dir).glob('*.md'))
        if not md_files:
            print(f"No markdown file found in: {paper_output_dir}")
            continue

        # 使用第一个找到的md文件
        md_file = md_files[0]

        # 构建对应的metadata JSON文件路径
        json_filename = f"{paper_id}.json"
        json_filepath = os.path.join(metadata_dir, json_filename)

        # 检查metadata JSON文件是否存在
        if not os.path.exists(json_filepath):
            print(f"Metadata file does not exist: {json_filepath}")
            continue

        try:
            # 读取markdown内容
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # 读取metadata JSON
            with open(json_filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 添加content字段
            metadata['content'] = md_content

            # 添加主题信息
            metadata['topic'] = topic_dir.split('_', 1)[1].replace('_', ' ')

            # 将论文数据添加到列表
            all_papers.append(metadata)

            print(f"Processed: {paper_id}")

        except Exception as e:
            print(f"Error processing {paper_id}: {str(e)}")

# 创建pandas DataFrame
papers_df = pd.DataFrame(all_papers)

# 显示基本信息
print("\nDataFrame created with shape:", papers_df.shape)
print("\nColumns:", papers_df.columns.tolist())

# # 保存DataFrame到CSV文件（可选）
# papers_df.to_csv("papers/papers_metadata.csv", index=False)
# print("\nDataFrame saved to papers_metadata.csv")

# 也可以保存为pickle文件以保留所有数据类型和结构
papers_df.to_pickle("papers/papers_metadata.pkl")
print("DataFrame saved to papers_metadata.pkl")

# 保存DataFrame到Parquet文件
papers_df.to_parquet("papers/papers_metadata.parquet", index=False)
print("\nDataFrame saved to papers_metadata.parquet")