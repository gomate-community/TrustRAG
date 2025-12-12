#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
应用基础配置。
"""
from api.rag.apps.config import settings
class AppConfig:
    """配置类"""
    API_V1_STR: str = ""
    # 文档接口描述相关的配置
    DOCS_URL = API_V1_STR + '/docs'
    REDOC_URL = API_V1_STR + '/redocs'
    OPENAPI_URL = API_V1_STR + '/openapi_url'
    API_VERSION = "v1"

    TITLE = "FASTAPI 模板函数"

    DESC = """

           """
    TAGS_METADATA = [
    ]
    # 配置代理相关的参数信息
    SERVERS = [
        {"url": "/", "description": "开发接口地址"},
        {"url": "/v2", "description": "测试地址"},
    ]

    WEB_URL: str = '*'
    # 接口地址
    API_URL: str = settings.api_url
    # 运行访问的地址
    API_HOST: str = settings.api_host
    # 端口
    API_PORT: int = settings.api_port

    DEBUGGER: bool = True

    SHOW_DOCS: bool = True

