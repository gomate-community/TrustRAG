import pandas as pd
import numpy as np
from trustrag.modules.engine.qdrant import QdrantEngine
from trustrag.modules.retrieval.embedding import SentenceTransformerEmbedding
from tqdm import tqdm

# Load the dataset
papers_df = pd.read_parquet("papers/papers_metadata.parquet")

# Initialize the embedding model
local_embedding_generator = SentenceTransformerEmbedding(
    model_name_or_path="G:/pretrained_models/mteb/bge-m3",
    device="cuda"
)
print(local_embedding_generator.model)

# Initialize QdrantEngine
qdrant_engine = QdrantEngine(
    collection_name="arxiv_llms",
    embedding_generator=local_embedding_generator,
    qdrant_client_params={"host": "192.168.1.5", "port": 6333},
    vector_size=local_embedding_generator.embedding_size
)
def search_example():
    # Example search query
    search_results = qdrant_engine.search(text="Chain of thought", limit=5)
    print("\nExample search results:")
    for result in search_results:
        # dict_keys(['payload', 'score'])
        print(result.keys())
        print(result)
def encode_paper():
    # Process the papers in batches to avoid memory issues
    batch_size = 10  # Adjust based on your GPU memory
    total_batches = len(papers_df) // batch_size + (1 if len(papers_df) % batch_size > 0 else 0)

    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(papers_df))

        batch_df = papers_df.iloc[start_idx:end_idx]

        # Combine title, summary, and content for each paper
        texts = []
        payloads = []

        for _, paper in batch_df.iterrows():
            # Create a combined text for embedding
            combined_text = f"Title: {paper['title']}\nSummary: {paper['summary']}\n"
            if 'content' in paper and pd.notna(paper['content']):
                # Add some content, but limit it to avoid too long inputs
                content = paper['content']
                if isinstance(content, str) and len(content) > 0:
                    combined_text += f"Content: {content[:1000]}..."  # Truncate long content

            texts.append(combined_text)

            paper_payload = paper.to_dict()
            # Process each field in the payload to ensure it's serializable
            for key in list(paper_payload.keys()):
                # Convert numpy arrays to lists
                if isinstance(paper_payload[key], np.ndarray):
                    paper_payload[key] = paper_payload[key].tolist()
                # Convert timestamps to ISO format strings
                elif isinstance(paper_payload[key], pd.Timestamp):
                    paper_payload[key] = paper_payload[key].isoformat()
                # Convert other numpy types to Python native types
                elif isinstance(paper_payload[key], (np.integer, np.floating)):
                    paper_payload[key] = paper_payload[key].item()
            payloads.append(paper_payload)

        # Generate embeddings for the batch
        vectors = local_embedding_generator.generate_embeddings(texts)
        print(vectors)
        print(vectors.shape)
        print(type(vectors))
        print(vectors.dtype)
        # Upload vectors and payload to Qdrant
        qdrant_engine.upload_vectors(vectors=vectors, payload=payloads)

        print(f"Processed batch {batch_idx + 1}/{total_batches}, {end_idx}/{len(papers_df)} papers")

    print("All papers have been processed and stored in Qdrant")

