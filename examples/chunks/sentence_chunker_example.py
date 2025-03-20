from trustrag.modules.chunks.sentence_chunk import SentenceChunker

if __name__ == '__main__':
    with open("../../data/docs/news.txt","r",encoding="utf-8") as f:
        content=f.read()
    print(content)

    cc=SentenceChunker(chunk_size=64)

    chunks=cc.get_chunks([content])
    print(chunks)