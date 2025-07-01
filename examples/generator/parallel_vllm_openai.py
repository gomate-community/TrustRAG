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
中美关税战按下“暂停键”：金价跳水 A股三大股指集体高开 
2025-05-13 09:48 发布于：北京市
【资料来源于新华网、玉渊谭天、大象新闻、第一财经】

北京时间5月12日下午15时，A股收盘之后，《中美日内瓦经贸会谈联合声明》正式公布。中美双方承诺于5月14日前采取举措，同步调整关税税率。此消息一出，中国股民很兴奋，同时美股三大指数全线大涨，纳指盘初暴涨超4%，中国资产集体爆发，纳斯达克中国金龙指数盘初大涨超5%。另外，被称为“恐慌指数”的VIX指数期货一度大跌超12%。

5月13日，A股三大股指集体高开，其中，沪指涨0.5%报3386.23点，深成指涨0.98%报10401.95点，创指涨1.29%报2091.35点。消费电子、跨境电商、固态电池、新能源车、机器人、半导体、华为鸿蒙、AI应用、算力题材活跃；黄金股连续回调。Wind统计显示，两市及北交所共4747只股票上涨，385只股票下跌，平盘有278只股票。

看懂中美经贸高层会谈联合声明 最终加征多少？

关税新政生效后，中美双边加征关税水平到底是多少？本次降幅有多大？据媒体玉渊谭天消息，当地时间5月12日，中美日内瓦经贸会谈联合声明发布，其中两方都提到，24%的关税在初始的90天内暂停实施，同时保留对这些商品加征剩余10%的关税。这里提到的24%和10%，是当地时间4月2日，美国对中国加征的所谓34%的“对等关税”。这34%中，有10%是所谓的基准关税，24%是对中国特别加征。复旦大学美国研究中心副主任宋国友告诉谭主，这意味着美国取消了对中国的特殊歧视。而在34%的关税之外，美国还曾先后对中国加征50%以及41%的所谓“对等关税”，中方都予以反制。

简而言之，当前美方取消了共计91%的对华加征关税，暂停实施24%的“对等关税”，保留剩余10%的关税。中方也相应取消和暂停了相同水平的关税。
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
