import openai

client = openai.Client(
  api_key="api-key",
  base_url="http://localhost:9997/v1"
)
response=client.embeddings.create(
  model="bge-m3",
  input=["What is the capital of China?"]
)
print(type(response.data[0].embedding),len(response.data[0].embedding),response.data[0].embedding,)
# <class 'list'> 1024 [-0.031030284240841866, ]