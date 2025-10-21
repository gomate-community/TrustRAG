import openai

client = openai.Client(
  api_key="cannot be empty",
  base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
)
client.embeddings.create(
  model="bge-m3",
  input=["What is the capital of China?"]
)