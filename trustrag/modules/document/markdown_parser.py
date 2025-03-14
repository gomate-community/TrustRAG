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
    ) -> List[str]:
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

    def merge_header_contents(self, documents: List[Document]) -> List[str]:
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

        return paragraphs,merged_data