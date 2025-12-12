import json
import re
from typing import List, Any

from openai import OpenAI
from trustrag.modules.rewriter.base import BaseRewriter


class OpenaiRewriterConfig:
    """Config for OpenAI-compatible rewriter."""

    def __init__(self, base_url: str, api_key: str | None = None, model_name: str | None = None, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def log_config(self):
        return f"""
        OpenaiRewriterConfig:
            Base URL: {self.base_url}
            API Key: {'*' * 8 if self.api_key else 'Not Set'}
            Model Name: {self.model_name}
            Timeout: {self.timeout}s
        """


class OpenaiRewriter(BaseRewriter):
    """
    A Rewriter that calls an OpenAI-compatible chat completion endpoint.
    """

    def __init__(self, config: OpenaiRewriterConfig):
        super().__init__()
        self.config = config
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "",
            timeout=self.config.timeout,
        )
        self.model_name = self.config.model_name
        print('Successful Init ChatGPT Rewriter ')

    def repair_json_output(self,content: str) -> str:
        """
        Repair and normalize JSON output.

        Args:
            content (str): String content that may contain JSON

        Returns:
            str: Repaired JSON string, or original content if not JSON
        """
        content = content.strip()
        if content.startswith(("{", "[")) or "```json" in content or "```ts" in content:
            try:
                # If content is wrapped in ```json code block, extract the JSON part
                if content.startswith("```json"):
                    content = content.removeprefix("```json")

                if content.startswith("```ts"):
                    content = content.removeprefix("```ts")

                if content.endswith("```"):
                    content = content.removesuffix("```")

                # Try to repair and parse JSON
                repaired_content = json.loads(content)
                return json.dumps(repaired_content, ensure_ascii=False)
            except Exception as e:
                print(f"JSON repair failed: {e}")
        return content

    def parse_response(self, response_data: str):
        """
        解析JSON响应字符串，如果解析失败则返回默认空值
        Args:
            json_string (str): JSON格式的响应字符串

        Returns:
            dict: 包含location、date和event字段的字典，解析失败时返回空值
        """
        # 定义默认的返回结构
        default_response = {
            "location": "",
            "date": "",
            "event": ""
        }

        try:
            # 如果response是字符串，则需要再次解析
            if isinstance(response_data, str):
                try:
                    # response_data = re.sub(r'^.*?```json\n|```$', '', response_data, flags=re.DOTALL)
                    response_data=self.repair_json_output(response_data)
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    print("报错",response_data)
                    return default_response

            # 从解析后的数据中提取字段，如果不存在则使用空字符串
            result = {
                "location": response_data.get("location", ""),
                "date": response_data.get("date", ""),
                "event": response_data.get("event", "")
            }

            return result

        except json.JSONDecodeError:
            print("报错")
            return default_response
        except Exception:
            print("报错")
            return default_response

    def rewrite(self, query: str) -> dict:
        system_prompt = """
请分析用户问题并提取其中的地点、时间、活动或会议名称，将这些信息以JSON格式输出。如果信息不全或用户未提及，则标记为""。按以下格式生成JSON输出：
        {
            "location": "地点或国家名称（如有）",
            "date": "时间日期或者时间范围（比如最新或最近等）（如有）",
            "event": "活动或会议名称（如有）"
        }

        示例：
        输入："在“一带一路”国际合作高峰论坛上，习近平讲了什么？"
        输出：{
            "location": "",
            "date": "",
            "event": "一带一路国际合作高峰论坛"
        }

        输入："在全国卫生与健康大会上，习近平对医疗卫生服务体系改革有哪些具体部署？"
        输出：{
            "location": "",
            "date": "",
            "event": "全国卫生与健康大会"
        }

        输入："在中央外事工作会议上，习对对外工作和外交战略有哪些具体部署？"
        输出：{
            "location": "",
            "date": "",
            "event": "中央外事工作会议"
        }

        输入："习近平在福建考察有什么重要指示？"
        输出：{
            "location": "福建",
            "date": "",
            "event": ""
        }
        
        输入："习近平在福建考察有什么重要指示？"
        输出：{
            "location": "福建",
            "date": "",
            "event": ""
        }
        输入："习近平关于新质生产力有什么最新的论述？"
        输出：{
            "location": "",
            "date": "最新",
            "event": ""
        }
        
        用户问题：
        """

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": system_prompt+query},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content
        return self.parse_response(content)

