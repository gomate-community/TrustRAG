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
from api.rag.apps.core.rewrite.bodys import RewriteBody
from api.rag.apps.handle.response.json_response import ApiResponse
from trustrag.modules.rewriter.openai_rewrite import OpenaiRewriter, OpenaiRewriterConfig


rewriter_router = APIRouter()

rewriter_config = OpenaiRewriterConfig(
    base_url=settings.rewriter_api_url,
    api_key=settings.gomall_api_key,
    model_name=settings.llm_name,
)
openai_rewriter = OpenaiRewriter(rewriter_config)

# Create
@rewriter_router.post("/rewrite/", response_model=None, summary="改写查询")
async def rewrite(rewrite_body: RewriteBody):
    query = rewrite_body.query
    response = openai_rewriter.rewrite(query)
    return ApiResponse(response, message="改写查询")
