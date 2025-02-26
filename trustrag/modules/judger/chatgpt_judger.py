import time
from typing import List, Any

import requests
from tqdm import tqdm

from trustrag.modules.judger.base import BaseJudger


class OpenaiJudgerConfig:
    """LLM 判断器配置类
    
    用于配置 LLM 判断器的服务参数，包括 API URL、密钥等
    """

    def __init__(self, base_url=None, api_key=None, model_name=None):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

    def log_config(self):
        # 记录当前配置设置
        return f"""
        OpenaiJudgerConfig:
            Base URL: {self.base_url}
            API Key: {'*' * 8 if self.api_key else 'Not Set'}
            Model Name: {self.model_name}
        """


class OpenaiJudger(BaseJudger):
    """使用 LLM 的判断器
    
    通过调用 LLM API 服务来判断文档与查询的相关性
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model_name = config.model_name
        print('成功初始化 ChatGPT 判断器')

    def judge(self, query: str, documents: List[str], k: int = 5, is_sorted: bool = False) -> list[dict[str, Any]]:
        system_prompt = """
        判断给定文章是否回答了查询，请严格遵循以下步骤并返回结果 1 或 0：

        1. 会议或活动名称检查：
           - 如果查询包含会议或活动的名称，请检查文章标题是否提到会议相关名称。
             - 若标题提到该名称，返回 1；若未提到该名称，返回 0。
             - 如果查询中出现会议或者活动名称，需要严格遵守上面规则，不确定是否提及名称返回0

        2. 相关性检查：
           - 如果查询中没有会议或活动名称，则判断文章内容是否与查询相关。
             - 若内容相关，返回 1；若不相关，返回 0。

        注意：只返回 1 或 0，不解释原因，不输出其他内容。
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        results = []
        for doc in tqdm(documents, desc="判断文档相关性"):
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"查询：{query}\n\n文章：{doc}"}
                ]
            }

            try:
                response = requests.post(
                    self.base_url + "/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                score = float(result['choices'][0]['message']['content'].strip())
                results.append({"text": doc, "score": score})
            except Exception as e:
                print(f"调用 LLM 服务失败: {str(e)}")
                results.append({"text": doc, "score": 0.0})
            time.sleep(0.1)  # 避免请求过快

        if is_sorted:
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            if k > 0:
                results = results[:k]

        return results
