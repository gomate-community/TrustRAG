from openai import OpenAI

#
# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
# )
#
# completion = client.chat.completions.create(
#     model="DeepSeek-R1-Distill-Qwen-1.5B",
#     messages=[
#         {"role": "user", "content": "请介绍下中科院计算所<think>\n"}
#     ],
#     temperature=0.7,       # 适中随机性
#     top_p=0.9,             # 控制生成词汇的分布
#     max_tokens=300,        # 限制输出长度
#     frequency_penalty=0.1, # 轻微降低重复词
#     presence_penalty=0.6   # 适当鼓励新内容
# )
#
# print(completion.choices[0].message)


# from openai import OpenAI
#
# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
# )
#
# completion = client.chat.completions.create(
#     model="DeepSeek-R1-Distill-Qwen-1.5B",
#     messages=[
#         {"role": "user", "content": "请介绍下中科院计算所<think>\n"}
#     ],
#     temperature=0.7,       # 适中随机性
#     top_p=0.9,             # 控制生成词汇的分布
#     max_tokens=300,        # 限制输出长度
#     frequency_penalty=0.1, # 轻微降低重复词
#     presence_penalty=0.6   # 适当鼓励新内容
# )
#
# print(completion.choices[0].message)


# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="sk-xxx",
# )
#
# stream = client.chat.completions.create(
#     model="DeepSeek-R1-Distill-Qwen-1.5B",
#     messages=[
#         {"role": "user", "content": "请介绍下中科院计算所"}
#     ],
#     stream=True
# )
#
# # Process the streaming response
# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")


# client = OpenAI(
#     base_url="http://10.208.63.29:8888/v1",
#     api_key="sk-xxx",
# )
#
# completion = client.chat.completions.create(
#     model="GLM4-9B-Chat",
#     messages=[
#         {"role": "user", "content": "请介绍下中科院计算所"}
#     ],
#     stream=False
# )
#
#
# print(completion)
# print(completion.choices[0].message)


client = OpenAI(
    base_url="http://10.208.62.156:7000/v1",
    api_key="sk-xxx",
)

response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {"role": "user", "content": "请介绍下中科院计算所"}
    ],
    stream=False
)

print(response)
print(response.choices[0].message)

# 当触发深度思考时，打印思维链内容
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)

client = OpenAI(
    api_key="sk-xxx",  # 随便填写，只是为了通过接⼝参数校验
    base_url="http://10.208.62.156:7000/v1",
)
response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {"role": "user", "content": "请写⼀⾸关于秋天的诗。"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=4096,
    presence_penalty=0.5,
    frequency_penalty=0.8,
    stream=False,
)
# 当触发深度思考时，打印思维链内容
if hasattr(response.choices[0].message, 'reasoning_content'):
    print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)
