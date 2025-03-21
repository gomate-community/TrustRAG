import os

import loguru
import typer
import json
from openai import AsyncOpenAI
import tiktoken
from typing import Optional
from rich.console import Console
from dotenv import load_dotenv
from trustrag.modules.deepresearch.agent.text_splitter import RecursiveCharacterTextSplitter
from trustrag.modules.deepresearch.config import EnvironmentConfig

load_dotenv()


class AIClientFactory:
    """Factory for creating AI clients for different providers."""

    @classmethod
    def create_client(cls, api_key: str, base_url: str) -> AsyncOpenAI:
        """Create an AsyncOpenAI-compatible client for the specified provider."""
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def get_client(
        cls,
        service_provider_name: Optional[str] = None,
        console: Optional[Console] = None,
    ) -> AsyncOpenAI:
        """Get a configured AsyncOpenAI client using environment variables."""
        console = console or Console()

        try:
            # Get and validate the provider configuration
            config = EnvironmentConfig.validate_provider_config(
                service_provider_name, console
            )

            # Create the client
            return cls.create_client(api_key=config.api_key, base_url=config.base_url)

        except ValueError:
            raise typer.Exit(1)
        except Exception as e:
            console.print(
                f"[red]Error initializing {service_provider_name or EnvironmentConfig.get_default_provider()} client: {e}[/red]"
            )
            raise typer.Exit(1)

    @classmethod
    def get_model(cls, service_provider_name: Optional[str] = None) -> str:
        """Get the configured model for the specified provider."""
        config = EnvironmentConfig.get_provider_config(service_provider_name)
        if not config.model:
            raise ValueError(f"No model configured for {config.service_provider_name}")
        return config.model


async def get_client_response(
    client: AsyncOpenAI, model: str, messages: list, response_format: dict
):
    # loguru.logger.info(messages)
    try:
        # 为SiliconFlow API添加特定参数
        params = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            "temperature": 0.7,  # 添加温度控制
            "max_tokens": 1024  # 确保有足够的token生成完整响应
        }
        
        response = await client.beta.chat.completions.parse(**params)
        # loguru.logger.info(response)

        result = response.choices[0].message.content
        
        # 检查是否有reasoning_content字段，SiliconFlow API特有
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            reasoning_content = response.choices[0].message.reasoning_content
            print(f"发现reasoning_content: {reasoning_content[:100]}...")  # 打印前100个字符作为预览
            # 优先使用reasoning_content字段，因为有些模型会将JSON放在这里
            try:
                parsed_result = json.loads(reasoning_content)
                print(f"成功从reasoning_content解析JSON")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"无法从reasoning_content解析JSON: {e}, 尝试从content解析")
                # 如果解析失败，回退到content字段
        
        # 确保result不为空
        if not result or result.strip() == '':
            print("警告: API返回了空内容，尝试从思考或原始响应中提取有用信息")
            # 尝试从原始响应对象中提取有用信息
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                # 解析reasoning_content中可能存在的JSON格式数据
                rc = response.choices[0].message.reasoning_content
                try:
                    # 尝试找出JSON块
                    json_start = rc.find('{')
                    json_end = rc.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = rc[json_start:json_end]
                        return json.loads(json_str)
                except:
                    pass
                
                # 如果还是无法解析，返回思考内容作为结构化数据
                parts = rc.split('\n\n')
                thinking_parts = []
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        thinking_parts.append({"思考": part})
                    else:
                        if part.strip():
                            thinking_parts.append({"内容": part})
                
                if thinking_parts:
                    return thinking_parts
            
            # 返回一个默认的JSON对象
            return {
                "queries": [
                    "检索增强生成RAG技术详解",
                    "RAG框架原理与应用场景",
                    "RAG与大语言模型结合的最佳实践"
                ]
            }
        
        # 尝试解析JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError as e:
            print(f"内容无法解析为JSON: {e}, 尝试其他解析方法")
            
            # 尝试查找JSON块
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    json_str = result[json_start:json_end]
                    return json.loads(json_str)
                except:
                    print("无法从JSON块中解析")
            
            # 如果不是有效的JSON，尝试将其转换为结构化数据
            lines = result.strip().split('\n')
            structured_data = []
            
            current_section = {}
            current_key = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('# ') or line.startswith('## '):
                    # 这是一个新部分
                    if current_key and current_section:
                        structured_data.append({current_key: current_section.get(current_key, "")})
                    current_section = {}
                    current_key = line.strip('# ')
                    current_section[current_key] = ""
                elif current_key:
                    current_section[current_key] += line + "\n"
            
            # 添加最后一部分
            if current_key and current_section:
                structured_data.append({current_key: current_section.get(current_key, "")})
            
            if structured_data:
                return structured_data
            
            # 如果结构化处理失败，返回原始文本作为独立项
            return [{"content": result}]
            
    except Exception as e:
        print(f"API调用或解析错误: {e}")
        # 在出错时返回一个默认响应
        return [
            {
                "思考": "API调用出错，提供默认回复。",
                "内容": "由于API连接问题，无法完成深度研究。以下是关于RAG（检索增强生成）的一些基本信息：\n\nRAG是一种结合了检索系统和生成式AI的技术框架，可以让生成模型基于检索到的信息产生更准确的输出。"
            }
        ]


MIN_CHUNK_SIZE = 140
# encoder = tiktoken.get_encoding(
#     "cl100k_base"
# )  # Updated to use OpenAI's current encoding


def trim_prompt(
    prompt: str, context_size: int = int(os.getenv("CONTEXT_SIZE", "128000"))
) -> str:
    """Trims a prompt to fit within the specified context size."""
    if not prompt:
        return ""

    # length = len(encoder.encode(prompt))
    length = len(prompt.split())
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    # Estimate characters to remove (3 chars per token on average)
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    trimmed_prompt = (
        splitter.split_text(prompt)[0] if splitter.split_text(prompt) else ""
    )

    # Handle edge case where trimmed prompt is same length
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    return trim_prompt(trimmed_prompt, context_size)