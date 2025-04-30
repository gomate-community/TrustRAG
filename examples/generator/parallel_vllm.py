import asyncio
import aiohttp
import time
import json

# 模型 API 端点
API_URL = "http://192.168.30.10:8001/v1/chat/completions"

# 请求头部
HEADERS = {
    "Content-Type": "application/json"
}

# 并发请求的数量（总共发送多少个请求）
num_requests = 100

# 同时并发的请求数
concurrency = 100

# 请求 payload 模板
payload_template = {
    "model": "Qwen2.5-32B-Instruct",
    "messages": [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "你好，给我讲个笑话。"*1000}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": False
}


async def fetch(session, i):
    try:
        async with session.post(API_URL, headers=HEADERS, data=json.dumps(payload_template)) as response:
            result = await response.json()
            print(result)
            print(f"[{i}] 响应状态: {response.status}, 内容摘要: {str(result)[:100]}")
    except Exception as e:
        print(f"[{i}] 请求失败: {e}")


async def run():
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch(session, i) for i in range(num_requests)]
        start = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start
        print(f"\n完成 {num_requests} 次请求，总耗时: {duration:.2f} 秒，平均: {duration / num_requests:.2f} 秒/请求")


if __name__ == "__main__":
    asyncio.run(run())
