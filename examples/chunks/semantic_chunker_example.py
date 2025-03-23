from trustrag.modules.chunks.semantic_chunk import SemanticChunker
from trustrag.modules.retrieval.embedding import SentenceTransformerEmbedding

if __name__ == "__main__":
    # Load the text from the file
    with open('../../../data/docs/新浪新闻.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Create the embedding generator
    # If you're using a local model like in your original code, specify the path
    embedding_generator = SentenceTransformerEmbedding(
        model_name_or_path="G:/pretrained_models/mteb/bge-m3"  # Change to your model path if needed
    )

    chunker = SemanticChunker(embedding_generator)

    # Get chunks using different methods

    # Method 1: Using percentile thresholding (default)
    percentile_chunks = chunker.get_chunks(
        text=text,
        chunk_method="percentile",
        threshold=90,  # 90th percentile of distances
        buffer_size=1
    )

    # Method 2: Using absolute thresholding
    absolute_chunks = chunker.get_chunks(
        text=text,
        chunk_method="absolute",
        threshold=0.2,  # Cosine distance threshold of 0.2
        buffer_size=1
    )

    # Method 3: Using dynamic thresholding
    dynamic_chunks = chunker.get_chunks(
        text=text,
        chunk_method="dynamic",
        threshold=1.5,  # 1.5 standard deviations above the mean
        buffer_size=1
    )

    # Print the results
    print(f"Text was divided into {len(percentile_chunks)} chunks using percentile method")
    for i, chunk in enumerate(percentile_chunks):
        print(f"Chunk #{i + 1} ({len(chunk)} chars)")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print()

    # You can also print or process the chunks from other methods
    print(f"\nNumber of chunks using absolute threshold: {len(absolute_chunks)}")
    print(f"Number of chunks using dynamic threshold: {len(dynamic_chunks)}")
