from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': '怎么学习机器学习',
        }
    ],
    model='deepseek-r1:1.5b',
)


print(chat_completion)