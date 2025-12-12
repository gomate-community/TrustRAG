from openai import OpenAI

client = OpenAI(
    api_key="zhipu_key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# completion = client.chat.completions.create(
#     model="glm-4-air-0111",
#     messages=[
#         {"role": "system", "content": "你是一个智能客服助手，能够快速解答用户问题。"},
#         {"role": "user", "content": "我的订单为什么还没有发货？"}
#     ],
#     top_p=0.7,
#     temperature=0.9
# )
#
# print(completion.choices[0].message)
# print(completion.choices[0].message.content)

# completion = client.chat.completions.create(
#     model="glm-4-air-0111",
#     messages=[
#         {"role": "system", "content": "你是一个信息提取助手，能够从文本中提取关键信息。"},
#         {"role": "user", "content": "从以下文本中提取出人名、地点和时间：'昨天，张三在北京参加了AI技术大会。'"}
#     ],
#     top_p=0.7,
#     temperature=0.9
# )
#
# print(completion.choices[0].message)
# print(completion.choices[0].message.content)


# completion = client.chat.completions.create(
#     model="glm-4-air-0111",
#     messages=[
#         {"role": "system", "content": "你是一个情感分析助手，能够准确识别文本中的情感倾向。"},
#         {"role": "user", "content": "分析以下文本的情感：'今天的天气真好，心情特别愉快！'"}
#     ],
#     top_p=0.7,
#     temperature=0.9
# )
#
# print(completion.choices[0].message)
# print(completion.choices[0].message.content)


# completion = client.chat.completions.create(
#     model="glm-4-air-0111",
#     messages=[
#         {"role": "system", "content": "你是一个社交媒体文案创作助手，能够创作高质量的文案。"},
#         {"role": "user", "content": "帮我写一篇关于旅行的小红书文案，目的地是云南。"}
#     ],
#     top_p=0.7,
#     temperature=0.9
# )
#
# print(completion.choices[0].message)
# print(completion.choices[0].message.content)


# completion = client.chat.completions.create(
#     model="glm-4-air-0111",
#     messages=[
#         {"role": "system", "content": "你是一个SQL转换助手，能够将自然语言转换为SQL查询语句。"},
#         {"role": "user", "content": "查询所有年龄大于30岁的用户信息。只输出sql语句，不要输出其他内容"}
#     ],
#     top_p=0.7,
#     temperature=0.9
# )
#
# print(completion.choices[0].message)
# print(completion.choices[0].message.content)

completion = client.chat.completions.create(
    model="glm-4-air-0111",
    messages=[
        {"role": "system", "content": "你是一个思维导图生成助手，能够使用Mermaid格式生成思维导图。"},
        {"role": "user", "content": "生成一个关于AI技术的思维导图。"}
    ],
    top_p=0.7,
    temperature=0.9
)

print(completion.choices[0].message)
print(completion.choices[0].message.content)
