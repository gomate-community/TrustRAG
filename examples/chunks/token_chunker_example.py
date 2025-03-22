from trustrag.modules.chunks.token_chunk import TokenChunker

if __name__ == '__main__':
    # with open("../../../data/docs/news.txt","r",encoding="utf-8") as f:
    with open("../../data/docs/伊朗总统罹难事件.txt", "r", encoding="utf-8") as f:
        content = f.read()
    tc = TokenChunker(tokenizer_type="jieba")
    chunks = tc.get_chunks([content], chunk_size=128)
    for chunk in chunks:
        print(f"Chunk Content：\n{chunk}")
