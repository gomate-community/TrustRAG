from trustrag.modules.engine.qdrant import QdrantEngine
from trustrag.modules.vector.embedding import CustomServerEmbedding
if __name__ == "__main__":
    # Initialize embedding generators
    embedding_generator = CustomServerEmbedding(
        api_url= "http://10.208.63.29:6008/v1/embeddings",
        api_key= "sk-aaabbbcccdddeeefffggghhhiiijjjkkk",
        model_name= "bge-large-en-v1.5",
        timeout= 30,
        embedding_size=1024
    )
    qdrant_engine = QdrantEngine(
        collection_name="chunk_arxiv",
        embedding_generator=embedding_generator,
        qdrant_client_params={"host": "10.60.1.145", "port": 6333},
        vector_size=embedding_generator.embedding_size
    )

    results = qdrant_engine.search(text="Retrival Augmented Generation", limit=5)
    print(results)
    for result in results:
        print(result)