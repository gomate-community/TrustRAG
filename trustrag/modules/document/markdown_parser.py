#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: quincy qiang
@license: Apache Licence
@file: codeparser.py.py
@time: 2025/01/08
@contact: yanqiangmiffy@gmail.com
@software: PyCharm
@description: A custom Markdown parser for extracting and processing chunks from Markdown files.
"""
import re
from typing import List, Dict, Union
from trustrag.modules.document.utils import get_encoding
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents.base import Document




class MarkdownParser:
    """
    Custom Markdown parser for extracting and processing chunks from Markdown files.
    """

    def parse(
        self,
        fnm: Union[str, bytes],
        encoding: str = "utf-8",
    ) -> tuple[list[str], list]:
        """
        Extracts chunks of content from a given Markdown file.

        Args:
            fnm (Union[str, bytes]): The file path or byte stream of the Markdown file.
            encoding (str, optional): The encoding to use when reading the file. Defaults to "utf-8".

        Returns:
            List[str]: A list of merged paragraphs extracted from the Markdown file.
        """
        # If fnm is not a string (assumed to be a byte stream), detect the encoding
        if not isinstance(fnm, str):
            encoding = get_encoding(fnm) if encoding is None else encoding
            content = fnm.decode(encoding, errors="ignore")
            documents = self.parse_markdown_to_documents(content)
        else:
            loader = UnstructuredMarkdownLoader(fnm, mode="elements")
            documents = loader.load()
        paragraphs,merged_data = self.merge_header_contents(documents)
        return paragraphs,merged_data

    def parse_markdown_to_documents(self, content: str) -> List[Document]:
        """
        Parses a Markdown string into a list of Document objects.

        Args:
            content (str): The Markdown content to parse.

        Returns:
            List[Document]: A list of Document objects representing the parsed Markdown content.
        """
        # Regular expression to match Markdown headings
        heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)

        # Store the parsed results
        documents = []

        # Split the content into sections
        sections = content.split('\n')

        for section in sections:
            # Check if the section is a heading
            heading_match = heading_pattern.match(section)
            if heading_match:
                # Calculate the depth of the heading
                current_depth = len(heading_match.group(1)) - 1
                # Extract the heading content
                page_content = heading_match.group(2).strip()
                # Add to the results
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={"category_depth": current_depth}
                    )
                )
            else:
                # If not a heading and the content is not empty, add to the results
                if section.strip():
                    documents.append(
                        Document(page_content=section.strip(), metadata={})
                    )
        return documents

    def merge_header_contents(self, documents: List[Document]) -> tuple[list[str], list]:
        """
        Merges headers and their corresponding content into a list of paragraphs.

        Args:
            documents (List[Document]): A list of Document objects representing the parsed Markdown content.

        Returns:
            List[str]: A list of merged paragraphs, each containing a header and its corresponding content.
        """
        merged_data = []
        current_title = None
        current_content = []

        for document in documents:
            metadata = document.metadata
            category_depth = metadata.get('category_depth', None)
            page_content = document.page_content

            # If category_depth is 0, it indicates a top-level heading
            if category_depth == 0:
                # If current_title is not None, it means we have collected a complete heading and its content
                if current_title is not None:
                    # Merge the current title and content into a single string and add to merged_data
                    merged_content = "\n".join(current_content)
                    merged_data.append({
                        'title': current_title,
                        'content': merged_content
                    })
                    # Reset the current title and content
                    current_content = []

                # Update the current title and add Markdown heading markers based on category_depth
                current_title = f"{'#' * (category_depth + 1)} {page_content}"

            # If category_depth is not 0, it indicates body content or other headings
            else:
                # If current_title is None, it means the content starts with body text
                if current_title is None:
                    merged_data.append({
                        'title': '',
                        'content': page_content
                    })
                # Headings other than top-level (e.g., second-level, third-level, etc.)
                elif category_depth is not None:
                    # Add Markdown heading markers
                    current_content.append(f"{'#' * (category_depth + 1)} {document.page_content}")
                else:
                    # Add the content to the current content list
                    current_content.append(page_content)

        # Handle the last heading and its content
        if current_title is not None:
            merged_content = "\n".join(current_content)
            merged_data.append({
                'title': current_title,
                'content': merged_content
            })
        paragraphs = [item["title"] + "\n" + item["content"] for item in merged_data]
        merged_data=self.merge_data_entries(merged_data)
        return paragraphs,merged_data


    def merge_data_entries(self,data_list):
        """
        Merge data entries where content is empty with subsequent entries.

        Args:
            data_list (list): List of dictionaries with 'title' and 'content' keys

        Returns:
            list: Processed list with merged entries
        """
        if not data_list:
            return []

        result = []
        i = 0
        while i < len(data_list):
            current = data_list[i]

            # If content is not empty, add to result and move to next item
            if current["content"]:
                result.append(current.copy())
                i += 1
                continue

            # If content is empty, need to merge with subsequent entries
            merged_title = current["title"]
            merged_content = ""
            j = i + 1

            # Look ahead to find first non-empty content
            while j < len(data_list):
                next_item = data_list[j]
                merged_title += " " + next_item["title"].strip('# ')

                if next_item["content"]:
                    merged_content = next_item["content"]
                    break

                j += 1

            # Create merged entry
            result.append({
                "title": merged_title,
                "content": merged_content
            })

            # Skip all entries that were merged
            i = j + 1 if j < len(data_list) else len(data_list)

        return result



if __name__ == '__main__':
    data = [

        {
            "title": "# 《安全边际》",
            "content": "作者：塞思.卡拉曼\n导言\n第一章 投资者哪里最易出错1、投机者和失败的投资者2、与投资者对立的华尔街本质3、机构表现竞赛：客户是输家4、价值错觉：20 世纪 80 年代对垃圾债券的迷失和错误观念\n第二章 价值投资哲学5、明确你的投资目标6、价值投资：安全边际的重要性7、价值投资哲学起源8、企业评估艺术\n第三章 价值投资过程"
        },
        {
            "title": "# 导 言",
            "content": "投资者所采用的投资方法尽管种类繁多，但这些方法几乎很难带来长期成功，只会带来巨大的经济损失。它们中不具备合乎逻辑的投资程序，更像是投机或赌博。投资者经常经不住想赚快钱的诱惑，结果成了华尔街短暂疯狂的牺牲品。写本书的初衷有两个。"
        },
        {
            "title": "# 第一部分",
            "content": ""
        },
        {
            "title": "# 多数投资者会在哪里跌倒",
            "content": "投机者和失败的投资者\n与投资者对立的华尔街本质\n机构表现竞赛：客户是输家\n价值错觉：20 世纪 80 年代对垃圾债券的沉迷和错误观念"
        },
        {
            "title": "# 第一章 投机者和失败的投资者",
            "content": ""
        },
        {
            "title": "# 第二章 投机者和失败的投资者",
            "content": ""
        },
        {
            "title": "# 投资与投机的对比",
            "content": "马克•吐温说过：一个人的一生中在两种情况下他不应该去投机：当他输不起的时候，当他输得起的时候。正是由于如此，理解投资和投机之间的区别是取得投资成功的第一步。\n对投资者来说，股票代表的是相应企业的部分所有权，而债券则是给这些企业的贷款。投资者在比较证券的价格与他们被估测的价值之后做出买卖的决定。当他们认为自己知道一些其他人不知道、不关心或者宁愿忽略的事情时，他们就会买进交易。他们会买进那些回报与风险相比看起来有吸引力的证券，同时卖出那些回报不再能抵御风险的证券。。"
        }
    ]
    mp=MarkdownParser()
    result=mp.merge_data_entries(data)
    print(result)
