from trustrag.modules.chunks.sentence_chunk import SentenceChunker

if __name__ == '__main__':
    with open("../../data/docs/bbc新闻.txt","r",encoding="utf-8") as f:
        content=f.read()

    cc=SentenceChunker()

    chunks=cc.get_chunks([content],chunk_size=256)
    print(chunks)