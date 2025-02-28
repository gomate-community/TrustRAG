from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[
        {"role": "user", "content": "请介绍下中科院计算所<think>"}
    ]
)

print(completion.choices[0].message)



client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-xxx",
)

stream = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[
        {"role": "user", "content": "请介绍下中科院计算所"}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")