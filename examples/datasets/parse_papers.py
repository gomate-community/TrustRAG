import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from tqdm import tqdm

# 要处理的目录列表
directories = [
    "papers/topic_Chain_of_Thought/pdfs",
    "papers/topic_LLM_Post-Training/pdfs",
    "papers/topic_Reasoning_Large_Language_Models/pdfs",
]

def process_pdf(pdf_file_path, output_dir):
    pdf_file_name = os.path.basename(pdf_file_path)  # 获取 PDF 文件名
    name_without_suff = pdf_file_name.split(".")[0]  # 去掉文件扩展名

    # 准备环境
    local_image_dir = os.path.join(output_dir, "images")  # 图片输出目录
    local_md_dir = output_dir  # Markdown 输出目录
    image_dir = str(os.path.basename(local_image_dir))  # 图片目录名称

    os.makedirs(local_image_dir, exist_ok=True)  # 创建图片输出目录

    # 创建文件写入对象
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

    # 读取 PDF 文件字节
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_path)  # 读取 PDF 文件内容

    # 处理 PDF 文件
    # 创建数据集实例
    ds = PymuDocDataset(pdf_bytes)

    # 推断 PDF 文件类型并进行相应处理
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)  # 使用 OCR 进行解析
        pipe_result = infer_result.pipe_ocr_mode(image_writer)  # 处理 OCR 模式结果
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)  # 使用文本模式进行解析
        pipe_result = infer_result.pipe_txt_mode(image_writer)  # 处理文本模式结果

    # 绘制结果并获取内容
    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))  # 绘制模型结果
    model_inference_result = infer_result.get_infer_res()  # 获取模型推断结果
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))  # 绘制布局结果
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))  # 绘制跨度结果
    md_content = pipe_result.get_markdown(image_dir)  # 获取 Markdown 内容
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)  # 导出 Markdown 文件
    content_list_content = pipe_result.get_content_list(image_dir)  # 获取内容列表
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)  # 导出内容列表 JSON 文件
    middle_json_content = pipe_result.get_middle_json()  # 获取中间 JSON 内容
    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')  # 导出中间 JSON 文件

# 处理每个目录
for directory in directories:
    output_dir = os.path.join(directory, "output")  # 输出目录
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    for file_name in tqdm(os.listdir(directory)):
        if file_name.endswith(".pdf"):  # 检查文件是否为 PDF
            pdf_file_path = os.path.join(directory, file_name)  # 获取 PDF 文件路径
            process_pdf(pdf_file_path, output_dir)  # 处理 PDF 文件