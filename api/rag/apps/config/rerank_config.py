#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: rerank_config.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from api.rag.apps.config import settings

class RerankConfig:
    """重排序配置类"""

    model_name_or_path: str = settings.reranker_name
    base_url: str | None = settings.rerank_base_url
    api_key: str | None = settings.rerank_api_key
