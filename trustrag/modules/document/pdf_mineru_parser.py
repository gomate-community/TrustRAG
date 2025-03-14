import json

import requests

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.markdown_parser import MarkdownParser


class PdfParserWithMinerU:
    def __init__(self, url='http://localhost:8888/pdf_parse'):

        # 服务器URL
        self.url = url
        self.md_parser = MarkdownParser()

    def parse(self, pdf_file_path, output_dir: str = "output"):

        # PDF文件路径
        # pdf_file_path = 'path/to/your/file.pdf'
        print("正在基于MinerU解析pdf文件，请耐心等待，耗时时间较长。")
        # 请求参数
        params = {
            'parse_method': 'auto',
            'is_json_md_dump': 'true',
            'output_dir': output_dir
        }

        # 准备文件
        files = {
            'pdf_file': (pdf_file_path.split('/')[-1], open(pdf_file_path, 'rb'), 'application/pdf')
        }

        # 发送POST请求
        response = requests.post(self.url, params=params, files=files, timeout=2000)
        # 检查响应
        if response.status_code == 200:
            print("PDF解析成功")
            markdown_content = response.json()["markdown"]
            markdown_bytes = markdown_content.encode("utf-8")  # Convert string to bytes
            paragraphs, merged_data = self.md_parser.parse(markdown_bytes)
            return markdown_content, merged_data
        else:

            print(f"错误: {response.status_code}")
            print(response.text)
            return '', []


if __name__ == '__main__':
    pdf_parser = PdfParserWithMinerU(
        url='https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/gomatebackend/rag_dc/pdf_parse')
    # pdf_file_path= '../../../data/paper/16400599.pdf'
    pdf_file_path = '../../../data/docs/1737333890455-安全边际塞斯卡拉曼.pdf'
    markdown_content, merged_data = pdf_parser.parse(pdf_file_path)
    with open('../../../data/docs/1737333890455-安全边际塞斯卡拉曼.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    # print(merged_data)

    with open('../../../data/docs/1737333890455-安全边际塞斯卡拉曼.json', 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    tc = TextChunker()
    results = []
    for section in merged_data:
        chunks = tc.get_chunks([section['content']], chunk_size=512)
        results.extend(
            [{"subtitle": section['title'], "content": chunk} for chunk in chunks]
        )
    print(results)
