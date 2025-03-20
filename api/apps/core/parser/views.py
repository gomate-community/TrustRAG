#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: views.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import re
import mimetypes
import loguru
from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.docx_parser import DocxParser
from trustrag.modules.document.excel_parser import ExcelParser
from trustrag.modules.document.html_parser import HtmlParser
from trustrag.modules.document.json_parser import JsonParser
from trustrag.modules.document.pdf_parser_fast import PdfSimParser
from trustrag.modules.document.ppt_parser import PptParser
from trustrag.modules.document.txt_parser import TextParser

tc = TextChunker()
parse_router = APIRouter()


@parse_router.post("/parse/", response_model=None, summary="文件解析")
async def parser(file: UploadFile = File(...), chunk_size: int = 512):
    """解析上传的文件
    
    Args:
        file: 上传的文件
        chunk_size: 文本分块大小
        
    Returns:
        解析后的文本块
    """
    try:
        # 读取文件内容
        filename = file.filename
        content = await file.read()
        
        # 根据文件扩展名选择解析器
        if re.search(r"\.docx$", filename, re.IGNORECASE):
            parser = DocxParser()
        elif re.search(r"\.pdf$", filename, re.IGNORECASE):
            parser = PdfSimParser()
        elif re.search(r"\.xlsx?$", filename, re.IGNORECASE):
            parser = ExcelParser()
        elif re.search(r"\.pptx$", filename, re.IGNORECASE):
            parser = PptParser()
        elif re.search(r"\.txt$", filename, re.IGNORECASE):
            parser = TextParser()
        elif re.search(r"\.json$", filename, re.IGNORECASE):
            parser = JsonParser()
        elif re.search(r"\.html?$", filename, re.IGNORECASE):
            parser = HtmlParser()
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{filename}")

        # 解析文件
        loguru.logger.info(f"开始解析文件：{filename}")
        paragraphs = parser.parse(content)
        chunks = tc.get_chunks(paragraphs, chunk_size=chunk_size)
        return JSONResponse(
            content={
                "code": 200,
                "message": "文件解析成功",
                "data": {
                    "chunks": chunks,
                    "total": len(chunks)
                }
            }
        )
    except Exception as e:
        loguru.logger.error(f"文件解析失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"文件解析失败：{str(e)}")
