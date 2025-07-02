import copy
import json
import os
from typing import Optional, List

from loguru import logger
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from tqdm import tqdm


class MineruParser:
    """MineRU PDF解析器，支持单个和批量PDF处理"""

    def __init__(self, lang: List[str] = ['ch', 'en'], parse_method: str = "auto",
                 formula_enable: bool = True, table_enable: bool = True):
        """
        初始化MineruParser

        Args:
            lang: 支持的语言列表，默认为['ch', 'en']（中文和英文）
            parse_method: 解析方法，默认为"auto"
            formula_enable: 是否启用公式解析
            table_enable: 是否启用表格解析
        """
        self.lang = lang
        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable

    def process_single_pdf(self, pdf_path: str, output_dir: str,
                           generate_visualizations: bool = True,
                           target_lang: str = None) -> dict:
        """
        处理单个PDF文件

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            generate_visualizations: 是否生成可视化文件（layout.pdf, spans.pdf等）
            target_lang: 目标语言，如果为None则使用第一个支持的语言

        Returns:
            dict: 包含处理结果的字典
        """
        pdf_filename = os.path.basename(pdf_path)
        base_filename = os.path.splitext(pdf_filename)[0]

        # 确定使用的语言
        if target_lang is None:
            target_lang = self.lang[0] if isinstance(self.lang, list) else self.lang

        logger.info(f"正在处理PDF: {pdf_filename}，使用语言: {target_lang}")

        # 创建输出目录结构
        os.makedirs(output_dir, exist_ok=True)
        images_dir_path = os.path.join(output_dir, "images")
        images_dir_name = os.path.basename(images_dir_path)
        os.makedirs(images_dir_path, exist_ok=True)

        # 初始化文件写入器
        image_writer = FileBasedDataWriter(images_dir_path)
        md_writer = FileBasedDataWriter(output_dir)

        try:
            # 读取PDF内容
            pdf_bytes = read_fn(pdf_path)
            logger.info(f"从 {pdf_filename} 读取了 {len(pdf_bytes)} 字节")

            # 转换PDF字节
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)
            pdf_bytes_list = [new_pdf_bytes]
            p_lang_list = [target_lang]

            # 分析PDF
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list,
                p_lang_list,
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable
            )

            # 处理结果
            model_list = infer_results[0]
            model_json = copy.deepcopy(model_list)
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]

            # 生成中间JSON
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
            )

            pdf_info = middle_json["pdf_info"]

            # 生成markdown内容
            md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, images_dir_name)
            md_writer.write_string("content.md", md_content_str)
            logger.info(f"创建markdown文件: content.md")

            # 生成内容列表
            content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, images_dir_name)
            md_writer.write_string(
                "content_list.json",
                json.dumps(content_list, ensure_ascii=False, indent=4)
            )
            logger.info("创建内容列表JSON")

            # 生成中间JSON
            md_writer.write_string(
                "middle.json",
                json.dumps(middle_json, ensure_ascii=False, indent=4)
            )
            logger.info("创建中间JSON文件")

            # 生成可视化文件（可选）
            if generate_visualizations:
                # 绘制布局边界框
                draw_layout_bbox(pdf_info, new_pdf_bytes, output_dir, "layout.pdf")
                logger.info(f"创建布局可视化: layout.pdf")

                # 绘制span边界框
                draw_span_bbox(pdf_info, new_pdf_bytes, output_dir, "spans.pdf")
                logger.info(f"创建spans可视化: spans.pdf")

                # 保存原始PDF
                md_writer.write("model.pdf", new_pdf_bytes)
                logger.info(f"创建模型可视化: model.pdf")

            logger.info(f"成功处理 {pdf_filename}")

            return {
                "status": "success",
                "pdf_filename": pdf_filename,
                "output_dir": output_dir,
                "markdown_content": md_content_str,
                "content_list": content_list,
                "middle_json": middle_json,
                "used_language": target_lang
            }

        except Exception as e:
            logger.error(f"处理 {pdf_filename} 时出错: {str(e)}")
            return {
                "status": "error",
                "pdf_filename": pdf_filename,
                "error": str(e)
            }

    def process_batch_pdfs(self, pdfs_dir: str, output_base_dir: str,
                           generate_visualizations: bool = True,
                           skip_existing: bool = True,
                           target_lang: str = None) -> dict:
        """
        批量处理PDF文件

        Args:
            pdfs_dir: 包含PDF文件的目录
            output_base_dir: 输出基础目录
            generate_visualizations: 是否生成可视化文件
            skip_existing: 是否跳过已处理的文件
            target_lang: 目标语言，如果为None则使用第一个支持的语言

        Returns:
            dict: 包含批量处理结果的字典
        """
        if not os.path.exists(pdfs_dir):
            logger.error(f"目录未找到: {pdfs_dir}")
            return {
                "status": "error",
                "error": f"目录未找到: {pdfs_dir}"
            }

        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]
        total_pdfs = len(pdf_files)

        if total_pdfs == 0:
            logger.warning(f"在 {pdfs_dir} 中未找到PDF文件")
            return {
                "status": "warning",
                "message": f"在 {pdfs_dir} 中未找到PDF文件"
            }

        # 确定使用的语言
        if target_lang is None:
            target_lang = self.lang[0] if isinstance(self.lang, list) else self.lang

        logger.info(f"找到 {total_pdfs} 个PDF文件进行批量处理，使用语言: {target_lang}")

        processed_pdfs = 0
        failed_pdfs = 0
        skipped_pdfs = 0
        results = []

        # 创建输出基础目录
        os.makedirs(output_base_dir, exist_ok=True)

        for pdf_file in tqdm(pdf_files, desc="处理PDF文件"):
            base_filename = os.path.splitext(pdf_file)[0]
            pdf_path = os.path.join(pdfs_dir, pdf_file)
            output_dir = os.path.join(output_base_dir, base_filename)

            # 检查是否已处理（如果启用跳过选项）
            if skip_existing:
                md_file = os.path.join(output_dir, "content.md")
                if os.path.exists(md_file):
                    logger.info(f"PDF {pdf_file} 已处理，跳过...")
                    skipped_pdfs += 1
                    continue

            # 处理单个PDF
            result = self.process_single_pdf(
                pdf_path, output_dir, generate_visualizations, target_lang
            )

            results.append(result)

            if result["status"] == "success":
                processed_pdfs += 1
            else:
                failed_pdfs += 1

        logger.info(f"批量处理完成。已处理: {processed_pdfs}，失败: {failed_pdfs}，跳过: {skipped_pdfs}")

        return {
            "status": "complete",
            "total_pdfs": total_pdfs,
            "processed_pdfs": processed_pdfs,
            "failed_pdfs": failed_pdfs,
            "skipped_pdfs": skipped_pdfs,
            "results": results,
            "used_language": target_lang
        }

    def get_markdown_content(self, pdf_path: str, target_lang: str = None) -> Optional[str]:
        """
        快速获取PDF的markdown内容（不保存文件）

        Args:
            pdf_path: PDF文件路径
            target_lang: 目标语言，如果为None则使用第一个支持的语言

        Returns:
            str: markdown内容，如果处理失败返回None
        """
        try:
            # 确定使用的语言
            if target_lang is None:
                target_lang = self.lang[0] if isinstance(self.lang, list) else self.lang

            # 读取PDF内容
            pdf_bytes = read_fn(pdf_path)
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

            # 分析PDF
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [new_pdf_bytes],
                [target_lang],
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable
            )

            # 处理结果
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]

            # 生成中间JSON（使用临时写入器）
            temp_writer = FileBasedDataWriter("/tmp")
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, temp_writer, _lang, _ocr_enable, False
            )

            pdf_info = middle_json["pdf_info"]

            # 生成markdown内容
            md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, "images")

            return md_content_str

        except Exception as e:
            logger.error(f"从 {pdf_path} 获取markdown内容时出错: {str(e)}")
            return None


if __name__ == '__main__':
    # 初始化解析器，支持中文和英文
    parser = MineruParser(lang=['ch', 'en'], parse_method='auto')

    # 处理单个PDF（使用中文）
    result = parser.process_single_pdf(
        pdf_path="../temp/20250605-Qwen3 Embedding Advancing Text Embedding and.pdf",
        output_dir="../output/document",
        generate_visualizations=True,
        target_lang='ch'
    )

    # 批量处理PDF（使用英文）
    batch_result = parser.process_batch_pdfs(
        pdfs_dir="../temp/",
        output_base_dir="../output/",
        skip_existing=True,
        target_lang='en'
    )
