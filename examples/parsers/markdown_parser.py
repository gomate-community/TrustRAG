#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: pdfparser_example.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from trustrag.modules.document.markdown_parser import MarkdownParser

if __name__ == '__main__':
    parser=MarkdownParser()
    # paragraphs= parser.parse(fnm="../../data/docs/基础知识.md")
    # # print(chunks)
    # print(len(paragraphs))
    # for chunk in paragraphs:
    #     print("==="*10)
    #     print(chunk)

    _,paragraphs = parser.parse(fnm="../../output/pdf_parse/2503.04697v1.md")
    # print(chunks)
    print(len(paragraphs))
    for chunk in paragraphs:
        print("===" * 10)
        print(chunk)