import pandas as pd
import numpy as np
from trustrag.modules.engine.qdrant import QdrantEngine
from trustrag.modules.vector.embedding import SentenceTransformerEmbedding
from tqdm import tqdm

# Load the dataset
papers_df = pd.read_parquet("G:/datasets/arxiv/arxiv-metadata-oai-snapshot.parquet")
papers_df=papers_df[papers_df["categories"].str.contains("cs.")].reset_index(drop=True)
print(papers_df.shape)
# Initialize the embedding model
local_embedding_generator = SentenceTransformerEmbedding(
    model_name_or_path="G:/pretrained_models/mteb/bge-m3",
    device="cuda"
)
print(local_embedding_generator.model)

# Initialize QdrantEngine
qdrant_engine = QdrantEngine(
    collection_name="arxiv_kaggle_abstract",
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
    batch_size = 128  # Adjust based on your GPU memory
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
            texts.append(paper['title']+'\n'+paper['abstract'])

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
            authors_parsed = []
            for author in paper['authors_parsed']:
                authors_parsed.append(author.tolist())
            paper_payload["authors_parsed"]=authors_parsed
            payloads.append(paper_payload)
        vectors = local_embedding_generator.generate_embeddings(texts)
        # Upload vectors and payload to Qdrant
        qdrant_engine.upload_vectors(vectors=vectors, payload=payloads)

        print(f"Processed batch {batch_idx + 1}/{total_batches}, {end_idx}/{len(papers_df)} papers")
        # break
    print("All papers have been processed and stored in Qdrant")

encode_paper()
