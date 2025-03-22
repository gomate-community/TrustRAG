from trustrag.modules.chunks.char_chunk import CharChunker

if __name__ == '__main__':
    with open("../../data/docs/bbc新闻.txt","r",encoding="utf-8") as f:
        content=f.read()

    cc=CharChunker()

    chunks=cc.get_chunks([content],chunk_size=128)
    print(chunks)