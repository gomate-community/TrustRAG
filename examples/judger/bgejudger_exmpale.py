#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: main.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import pandas as pd
from trustrag.modules.judger.bge_judger import BgeJudger, BgeJudgerConfig

if __name__ == '__main__':
    judger_config = BgeJudgerConfig(
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
    )

    bge_judger = BgeJudger(judger_config)
    contexts = [
        '文档1',
        '发展新质生产力'
    ]
    judge_docs = bge_judger.judge(
        query="发展新质生产力",
        documents=contexts,
    )
    print(judge_docs)
