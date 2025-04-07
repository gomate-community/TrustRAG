#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: chat.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from abc import ABC
from openai import OpenAI
import openai
import re
from langdetect import detect

def is_english(texts):
    try:
        # 检测文本的语言
        return detect(str(texts[0])) == 'en'
    except:
        return False

class Base(ABC):
    def __init__(self, key, model_name, base_url):
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def chat(self, system, history, gen_conf={}):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                stream=True,
                **gen_conf)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:continue
                # ans += resp.choices[0].delta.content
                ans= resp.choices[0].delta.content
                total_tokens += 1
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens



class OpenAIChat(Base):
    def __init__(self, key, model_name="gpt-4o", base_url="https://www.dmxapi.com/v1"):
        if not base_url: base_url="https://www.dmxapi.com/v1"
        super().__init__(key, model_name, base_url)

class DeepSeekChat(Base):
    """
    https://platform.deepseek.com/usage
    """
    def __init__(self, key, model_name="deepseek-chat", base_url="https://api.deepseek.com"):
        if not base_url: base_url="https://api.deepseek.com"
        super().__init__(key, model_name, base_url)

class GPT_4o_up(Base):
    def __init__(self, key, model_name="gpt-4o", base_url="https://api.openai-up.com/v1"):
        if not base_url: base_url="https://api.openai-up.com/v1"
        super().__init__(key, model_name, base_url)

class GPT4_DMXAPI(Base):
    def __init__(self, key, model_name="gpt-4o-all", base_url="https://www.dmxapi.com/v1"):
        if not base_url: base_url="https://www.dmxapi.com/v1"
        super().__init__(key, model_name, base_url)

class GptTurbo(Base):
    def __init__(self, key, model_name="gpt-3.5-turbo", base_url="https://api.openai.com/v1"):
        if not base_url: base_url="https://api.openai.com/v1"
        super().__init__(key, model_name, base_url)


class MoonshotChat(Base):
    def __init__(self, key, model_name="moonshot-v1-8k", base_url="https://api.moonshot.cn/v1"):
        if not base_url: base_url="https://api.moonshot.cn/v1"
        super().__init__(key, model_name, base_url)


class XinferenceChat(Base):
    def __init__(self, key=None, model_name="", base_url=""):
        key = "xxx"
        super().__init__(key, model_name, base_url)

class ZhipuChat(Base):
    """
    https://www.bigmodel.cn/
    """
    def __init__(self, key, model_name="glm-4", base_url="https://open.bigmodel.cn/api/paas/v4/"):
        if not base_url: base_url="https://open.bigmodel.cn/api/paas/v4/"
        super().__init__(key, model_name, base_url)


class DashScopeChat(Base):
    """
    https://dashscope.aliyun.com/
    """
    def __init__(self, key, model_name="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        if not base_url: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        super().__init__(key, model_name, base_url)

class VolcengineChat(Base):
    """
    https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint
    """
    def __init__(self, key, model_name="deepseek-r1-250120", base_url="https://ark.cn-beijing.volces.com/api/v3"):
        if not base_url: base_url="https://ark.cn-beijing.volces.com/api/v3"
        super().__init__(key, model_name, base_url)


class HunYuanChat(Base):
    def __init__(self, key, model_name="hunyuan-t1-20250321", base_url="https://www.dmxapi.cn/v1"):
        if not base_url: base_url="https://www.dmxapi.cn/v1"
        super().__init__(key, model_name, base_url)

if __name__ == '__main__':


    # import os
    # from openai import OpenAI
    #
    # client = OpenAI(
    #     api_key="xx",
    #     base_url="https://ark.cn-beijing.volces.com/api/v3",
    # )
    #
    # # Non-streaming:
    # print("----- standard request -----")
    # completion = client.chat.completions.create(
    #     model="deepseek-r1-250120",  # your model endpoint ID
    #     messages=[
    #         {"role": "system", "content": "你是人工智能助手"},
    #         {"role": "user", "content": "常见的十字花科植物有哪些？"},
    #     ],
    # )
    # print(completion.choices)
    # print(completion.choices[0].message.content)

    # 初始化 DashScopeChat
    dashscope_chat = HunYuanChat(key="sk-hbMzKcHfnDKSU1NnJPDEFG4xOQE1kB2mcQpkJYtPqDTeTRcV")

    # 定义系统提示和用户消息
    system_message = "You are a helpful assistant."
    user_message = "你是谁？"

    # 调用 chat 方法
    response, total_tokens = dashscope_chat.chat(
        system=system_message,
        history=[{"role": "user", "content": user_message}],
        gen_conf={}  # 可以根据需要添加生成配置，如 top_p 和 temperature
    )

    # 打印响应
    print(response)
    print(f"Total tokens used: {total_tokens}")