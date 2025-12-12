import openai

# 设置OpenAI API密钥和基础URL
openai.api_key = "your_key"
openai.base_url = "https://www.dmxapi.com/v1/"

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 示例文本
text = ["这是一个示例文本,用于演示如何获取文本嵌入。","这是一个示例文本,用于演示如何获取文本嵌入。"]

# 获取文本嵌入
embedding = get_embedding(text)
print(type(embedding))
print(f"文本: {text}")
print(f"嵌入向量维度: {len(embedding)}")
print(f"嵌入向量前5个元素: {embedding[:5]}")
