# https://ai.google.dev/gemini-api/docs/openai?hl=zh_cn
# https://aistudio.google.com/app/apikey
from openai import OpenAI

client = OpenAI(
    api_key="xxx",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    n=1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "请帮我解释强化学习算法PPO"
        }
    ]
)

print(response.choices[0].message)