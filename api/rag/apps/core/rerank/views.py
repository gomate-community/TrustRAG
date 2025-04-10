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

from api.apps.core.rerank.bodys import RerankBody
from api.apps.core.rerank.models import Application
from api.apps.handle.response.json_response import UserNotFoundResponse, ApiResponse
from trustrag.modules.reranker.bge_reranker import BgeReranker, BgeRerankerConfig
from trustrag.config.config_loader import config

# from apps.handle.exception.exception import MallException
# from apps.core.config.models import LLMModel
# from tortoise.contrib.pydantic import pydantic_model_creator

rerank_router = APIRouter()
# 从配置文件加载重排序配置
rerank_service = config.get_config('services.rerank')
rerank_model = config.get_config('models.reranker')

reranker_config = BgeRerankerConfig(
    model_name_or_path=rerank_model['name'],
    api_key=rerank_service['api_key'],
    url=rerank_service['base_url']
)
bge_reranker = BgeReranker(reranker_config)
# Create
@rerank_router.post("/rerank/", response_model=None, summary="重排序检索文档")
async def rerank(rerank_body: RerankBody):
    contexts = rerank_body.contexts
    query=rerank_body.query
    top_docs = bge_reranker.rerank(
        query=query,
        documents=contexts,
        is_sorted=False
    )
    return ApiResponse(top_docs, message="重排序检索文档成功")
