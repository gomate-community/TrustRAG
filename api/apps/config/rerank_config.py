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
from trustrag.config.config_loader import config

class RerankConfig():
    """重排序配置类"""
    # 从配置文件加载服务和模型配置
    _rerank_service = config.get_config('services.rerank')
    _rerank_model = config.get_config('models.reranker')
    
    # 模型名称
    model_name_or_path:str = _rerank_model['name']
    # 服务 URL
    base_url:str = _rerank_service['base_url']
    # API 密钥
    api_key:str = _rerank_service['api_key']
