
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(

  model="deepseek/deepseek-chat-v3-0324:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices)
print(completion.choices[0].message.content)
