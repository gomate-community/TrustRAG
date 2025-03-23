from trustrag.modules.vector.embedding import SentenceTransformerEmbedding

embedding_generator = SentenceTransformerEmbedding(model_name_or_path="G:/pretrained_models/mteb/bge-m3")

text = "今天天气真好"

embedding = embedding_generator.generate_embedding(text)
print(embedding)



texts = ["今天天气真好","今天心情不错"]

embedding = embedding_generator.generate_embeddings(texts)
print(embedding)
