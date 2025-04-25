import sys
sys.path.append("G:\Projects\TrustRAG")
from trustrag.modules.engine.qdrant import QdrantEngine
from trustrag.modules.vector.embedding import SentenceTransformerEmbedding

if __name__ == "__main__":
    print("hello word")
    # Initialize embedding generators
    local_embedding_generator = SentenceTransformerEmbedding(
        model_name_or_path="G:/pretrained_models/mteb/bge-m3",
        device="cuda"
    )
    print(local_embedding_generator.model)
    # Initialize QdrantEngine with local embedding generator
    qdrant_engine = QdrantEngine(
        collection_name="arxiv_llms",
        embedding_generator=local_embedding_generator,
        qdrant_client_params={"host": "192.168.1.5", "port": 6333},
        vector_size=local_embedding_generator.embedding_size
    )

    results = qdrant_engine.search(text="Retrival Augmented Generation，RAG",  limit=5)
    for result in results:
        print(result)