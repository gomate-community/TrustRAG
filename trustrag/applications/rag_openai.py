#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: RagApplication.py
@time: 2024/05/20
@contact: yanqiangmiffy@gamil.com
"""
import os

import loguru

from trustrag.modules.citation.match_citation import MatchCitation
from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.generator.chat import OpenAIChat
from trustrag.modules.prompt.templates import CHAT_PROMPT_TEMPLATES
from trustrag.modules.judger.llm_judger import LLMJudger
from trustrag.modules.retrieval.dense_retriever import DenseRetriever
from trustrag.modules.vector.embedding import OpenAIEmbedding
from trustrag.modules.retrieval.web_retriever import DuckduckSearcher
from trustrag.modules.rewriter.llm_rewriter import LLMRewriter


class ApplicationConfig():
    def __init__(self):
        self.retriever_config = None
        self.rerank_config = None


class RagApplication():
    def __init__(self, config):
        self.config = config
        self.parser = CommonParser()

        self.embedding_generator = OpenAIEmbedding(
            base_url=self.config.retriever_config.base_url,
            api_key=self.config.retriever_config.api_key,
            embedding_model_name=self.config.retriever_config.embedding_model_name
        )

        self.retriever = DenseRetriever(self.config.retriever_config, self.embedding_generator)
        # self.reranker = BgeReranker(self.config.rerank_config)
        self.llm = OpenAIChat(key=self.config.api_key, model_name=self.config.model_name, base_url=self.config.base_url)
        self.system_prompt = "你是一个可信可靠的问答助手。"
        self.llm_rewriter = LLMRewriter(api_key=self.config.api_key,model_name=self.config.model_name,base_url=self.config.base_url)
        self.llm_judger = LLMJudger(api_key=self.config.api_key,model_name=self.config.model_name,base_url=self.config.base_url)
        self.mc = MatchCitation()
        self.tc = TextChunker()
        self.web_searcher = DuckduckSearcher(proxy=None, timeout=20)

    def init_vector_store(self):
        """
        """
        print("init_vector_store ... ")
        if not os.path.exists(self.config.docs_path):
            os.makedirs(self.config.docs_path)
        all_paragraphs = []
        all_chunks = []
        for filename in os.listdir(self.config.docs_path):
            file_path = os.path.join(self.config.docs_path, filename)
            try:
                paragraphs = self.parser.parse(file_path)
                all_paragraphs.append(paragraphs)
            except:
                pass
        print("chunking for paragraphs")
        for paragraphs in all_paragraphs:
            chunks = self.tc.get_chunks(paragraphs, 256)
            all_chunks.extend(chunks)
        self.retriever.build_from_texts(all_chunks)
        print("init_vector_store done! ")
        self.retriever.save_index(self.config.retriever_config.index_path)

    def load_vector_store(self):
        self.retriever.load_index(self.config.retriever_config.index_path)

    def add_document(self, file_path):
        try:
            chunks = self.parser.parse(file_path)
            for chunk in chunks:
                self.retriever.add_text(chunk)
            print("add_document done!")
            response = {
                "detail": file_path,
                "status": "completed",
            }
        except Exception as e:
            response = {
                "detail": file_path,
                "status": "failed",
            }
        return response

    def chat(self, question: str = '', top_k: int = 5):
        rewrite_query = self.llm_rewriter.rewrite(question)
        rewrite_query = "\n".join(f"{i + 1}. {query.strip()};" for i, query in enumerate(rewrite_query.split(";")))
        loguru.logger.info("Query Rewrite Results:" + rewrite_query)
        contents = self.retriever.retrieve(query=question+" "+rewrite_query, top_k=top_k)
        loguru.logger.info("Retrieve Results：")
        loguru.logger.info(contents)
        # contents = self.reranker.rerank(query=question, documents=[content['text'] for content in contents])
        documents = [content['text'] for content in contents]
        labels = self.llm_judger.judge(question, documents=documents)
        loguru.logger.info("Useful Judge Results:")
        for content, label in zip(contents, labels):
            content['label'] = label
        loguru.logger.info(contents)

        context_content = ""
        for idx, item in enumerate(contents):
            print(idx + 1)
            context_content = context_content + str(idx + 1) + "." + item['text'] + "\n"
        print(context_content)
        user_input = CHAT_PROMPT_TEMPLATES['RAG_PROMPT_TEMPALTE'].format(question=question, context=context_content)
        loguru.logger.info("User Request：\n" + user_input)

        history = [
            {"role": "user", "content": user_input}
        ]
        result, history = self.llm.chat(system=self.system_prompt, history=history, gen_conf={"temperature": 0.3})

        # 默认值定义
        selected_docs = []
        default_values = {
            "file_name": "default_file_name",
            "content": "default_content",
            "chk_id": -1,
            "doc_id": -1,
            "newsinfo": {
                "title": "default_title",
                "content": "default_newsinfo_content",
                "date": "default_date",
                "source": "default_source"
            }
        }
        evidences = [content['text'] for content in contents]
        # 遍历 evidences 并填充 selected_docs
        for idx, evidence in enumerate(evidences):
            doc = {
                "file_name": default_values["file_name"],
                "content": evidence,  # 将 evidence 内容赋值给 content 字段
                "chk_id": idx,  # 使用索引作为 chk_id
                "doc_id": idx + 1000,  # 假设 doc_id 从 1000 开始递增
                "newsinfo": {
                    "title": default_values["newsinfo"]["title"],
                    "content": evidence[:100],  # 截取前 100 字符作为新闻内容
                    "date": default_values["newsinfo"]["date"],
                    "source": default_values["newsinfo"]["source"]
                }
            }
            selected_docs.append(doc)

        result = self.mc.ground_response(
            question=question,
            response=result,
            evidences=[content['text'] for content in contents],
            selected_idx=[idx for idx in range(len(contents))],
            selected_docs=selected_docs,
            markdown=True
        )
        return result["result"], history, contents, rewrite_query
