from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

for key, value in os.environ.items():
    print(f"{key} = {value}")
client = OpenAI(
    # 替换为您需要调用的模型服务Base Url
    base_url=os.environ.get("BASE_URL"),
    # 环境变量中配置您的API Key
    api_key=os.environ.get("ARK_API_KEY")
)


print("----- standard request -----")
completion = client.chat.completions.create(
    model="deepseek-r1-250120",
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
)
print(completion)
print("reasoning_content:\n",completion.choices[0].message.reasoning_content)
print("------"*100)

print("content:\n")
print(completion.choices[0].message.content)




