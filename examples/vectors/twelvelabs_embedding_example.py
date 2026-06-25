"""
Multimodal video-RAG with TwelveLabs (Marengo embeddings + Pegasus analysis).

Marengo embeds text, image, audio and video into one shared 512-dim space, so a
text query can be matched against video clips directly. Pegasus turns a video
into searchable text. Get a free API key at https://twelvelabs.io and export it:

    export TWELVELABS_API_KEY="tlk_..."
"""
from trustrag.modules.vector.embedding import (
    EmbeddingFactory,
    TwelveLabsEmbedding,
    TwelveLabsVideoAnalyzer,
)

# --- Marengo text embeddings (512-dim, shared multimodal space) ---
embedding_generator = TwelveLabsEmbedding()  # or EmbeddingFactory.create_embedding_generator("twelvelabs")

text = "a cat playing the piano"
embedding = embedding_generator.generate_embedding(text)
print("dim:", len(embedding))  # 512

texts = ["a cat playing the piano", "a dog catching a frisbee"]
embeddings = embedding_generator.generate_embeddings(texts)
print("shape:", embeddings.shape)  # (2, 512)

# A text query and a video clip live in the same space, so you can rank videos
# against a text question with cosine_similarity from the base class:
#   image_vec = embedding_generator.embed_image("https://example.com/frame.jpg")
#   score = TwelveLabsEmbedding.cosine_similarity(embedding, image_vec)

# --- Pegasus video understanding (video -> text for grounding/retrieval) ---
analyzer = TwelveLabsVideoAnalyzer()
summary = analyzer.analyze(
    prompt="Summarize this video in two sentences.",
    video_url="https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
)
print("summary:", summary)
