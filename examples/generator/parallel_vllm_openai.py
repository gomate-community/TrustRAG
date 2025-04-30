import asyncio
import time
from openai import AsyncOpenAI

# 初始化 OpenAI 客户端，指向你的本地模型部署地址
client = AsyncOpenAI(
    base_url="http://192.168.30.10:8001/v1",
    api_key="NULL"  # 本地部署可用任意字符串
)

# prompt 模板
PROMPT_TEMPLATE = """请为以下新闻内容生成摘要总结，严格遵循要求：
1. **核心要素**：首句最好包含「5W1H」要素（Who-What-When-Where-Why-How）
2. **语言规范**：
   - 字数控制在130字以内
   - 使用主动语态（如"XX宣布"而非"被XX宣布"）
   - 避免主观形容词（"重大"/"惊人"等）
3. **禁止事项**：
   - 不得添加原文未提及的推论
   - 不得使用"本文报道"等元描述
4. **其他要求**：
   - 生成摘要内容必须是中文
   - 生成摘要必须是一大段话，不要分点
   - 直接输出生成的摘要内容，不要输出其他的内容

新闻原文：
\"\"\"{input_text}\"\"\"
"""

# 示例新闻（可扩展为多个并发测试文本）
news_text = """
这是一篇长新闻
"""

# 并发参数
model_name = "Qwen2.5-32B-Instruct"
num_requests = 1000
concurrency = 10

# 每个任务调用生成摘要
async def generate_summary(i, news):
    try:
        full_prompt = PROMPT_TEMPLATE.format(input_text=news.strip())
        start = time.time()
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        duration = time.time() - start
        output = response.choices[0].message.content.strip()
        print(f"[{i}] ✅ {duration:.2f}s: {output}")
    except Exception as e:
        print(f"[{i}] ❌ 请求失败: {e}")


async def main():
    sem = asyncio.Semaphore(concurrency)

    async def limited_task(i):
        async with sem:
            await generate_summary(i, news_text)

    tasks = [limited_task(i) for i in range(num_requests)]

    start_time = time.time()
    await asyncio.gather(*tasks)
    print(f"\n🌟 总耗时: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
