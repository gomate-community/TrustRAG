from typing import List
from trustrag.modules.generator.chat import OpenAIChat
class Decontextualizer:
    def __init__(self, api_key=None, base_url=None, model_name=None,
                 system_prompt="你是一名智能文本处理助手，专门用于去除文本中的指代。",
                 rewrite_prompt="请去除以下文本中的所有指代，并返回修改后的完整文本：\n\"{text}\""):
        """
        初始化 Decontextualizer 对象。
        :param api_key: 语言模型 API 密钥
        :param base_url: 语言模型 API 基础 URL
        :param model_name: 使用的语言模型名称
        :param system_prompt: 系统提示，定义任务背景
        :param rewrite_prompt: 重写提示模板，用于具体任务
        """
        self.api_key = api_key
        self.chat = OpenAIChat(key=api_key, base_url=base_url, model_name=model_name)
        self.system_prompt = system_prompt
        self.rewrite_prompt = rewrite_prompt

    def decontextualize(self, text: str) -> str:
        """
        去除输入文本中的指代。
        :param text: 输入的原始文本
        :return: 去指代后的文本
        """
        # 构造提示
        prompt = self.rewrite_prompt.format(text=text)

        # 构造对话历史
        history = [
            {"role": "user", "content": prompt}
        ]

        # 配置生成参数
        gen_conf = {
            "temperature": 0.7,  # 温度设置为稍高，以鼓励多样性
            "max_tokens": 500,   # 最大生成长度
        }

        # 调用语言模型生成结果
        response, _ = self.chat.chat(system=self.system_prompt, history=history, gen_conf=gen_conf)

        # 返回生成的文本
        return response.strip()

# 示例使用
if __name__ == '__main__':
    # 初始化 Decontextualizer
    decontextualizer = Decontextualizer(api_key='sk-gDbFoQAYz9pwqBsH0aPA1H8DN9s0B9F3vWNjjPcijRBFjk7f')

    # 输入文本
    input_text = """
    伊朗总统莱希在坠机事故中罹难。事故发生后，哈梅内伊表示希望莱希平安回家。然而，最终确认莱希不幸遇难。
    """

    # 去指代处理
    output_text = decontextualizer.decontextualize(input_text)
    print("去指代后的文本：")
    print(output_text)