#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: views.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import loguru
from fastapi import APIRouter

from api.rag.apps.config import settings
from api.rag.apps.core.judge.bodys import JudgeBody
from api.rag.apps.handle.response.json_response import ApiResponse
from trustrag.modules.judger.bge_judger import BgeJudger, BgeJudgerConfig
from trustrag.modules.judger.chatgpt_judger import OpenaiJudger, OpenaiJudgerConfig

judge_router = APIRouter()

# BGE 判断器配置
judge_config = BgeJudgerConfig(
    model_name_or_path=settings.reranker_name,
)
bge_judger = BgeJudger(judge_config)

# LLM 判断器配置
judger_config = OpenaiJudgerConfig(
    base_url=settings.gomall_base_url,
    api_key=settings.gomall_api_key,
    model_name=settings.llm_name,
)
openai_judger = OpenaiJudger(judger_config)


# Create
@judge_router.post("/judge/", response_model=None, summary="判断文档相关性")
async def judge(judge_body: JudgeBody):
    """判断文档相关性
    
    Args:
        judge_body: 包含查询和文档的请求体
        
    Returns:
        判断结果，包含相关性得分
    """
    contexts = judge_body.contexts
    query = judge_body.query
    method = judge_body.method

    if method == 'llm':
        loguru.logger.info("使用 LLM 进行判断...")
        judge_docs = openai_judger.judge(
            query=query,
            documents=contexts,
            is_sorted=False
        )
    else:
        loguru.logger.info("使用 BGE 进行判断...")
        judge_docs = bge_judger.judge(
            query=query,
            documents=contexts,
            is_sorted=False
        )
    return ApiResponse(judge_docs, message="文档相关性判断完成")
