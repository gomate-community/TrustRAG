from openai import OpenAI

client = OpenAI(
    base_url='http://192.168.1.5:11434/v1/',
    # base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)


# Streaming request example
stream = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': '怎么学习机器学习',
        }
    ],
    model='deepseek-r1:1.5b',
    stream=True,
)

# Process the streaming response
print("\n--- Streaming Response ---")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n--------------------------")

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


