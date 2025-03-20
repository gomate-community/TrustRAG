#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: textparser_exmaple.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from trustrag.modules.document.txt_parser import TextParser
from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.chunks.sentence_chunk import SentenceChunker


if __name__ == '__main__':
    text_parser=TextParser()
    tc=TextChunker()
    sc=SentenceChunker(chunk_size=512)
    paragraphs = text_parser.parse(fnm="../../data/docs/1737765690374-穷查理宝典.pdf-15a72b24-cc5c-4a4e-ae9e-7514e0d9be02.txt")
    print(len(paragraphs))
    chunks=tc.get_chunks(paragraphs,chunk_size=128)
    # chunks=sc.get_chunks(paragraphs)
    print(len(chunks))
    # for chunk in chunks:
    #     print(len(chunk))