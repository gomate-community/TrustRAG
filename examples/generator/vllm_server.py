from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[
        {"role": "user", "content": "请介绍下中科院计算所<think>\n"}
    ],
    temperature=0.7,       # 适中随机性
    top_p=0.9,             # 控制生成词汇的分布
    max_tokens=300,        # 限制输出长度
    frequency_penalty=0.1, # 轻微降低重复词
    presence_penalty=0.6   # 适当鼓励新内容
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