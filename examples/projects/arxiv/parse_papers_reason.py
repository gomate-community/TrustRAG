import logging
import os
import sys
from trustrag.modules.document.pdf_mineru_parser import MineruParser
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pdf_processor')


def process_pdf(pdf_path, output_dir):
    """
    Process a PDF file using MineruParser and generate various output files.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory where outputs will be saved
    """
    pdf_filename = os.path.basename(pdf_path)
    base_filename = os.path.splitext(pdf_filename)[0]

    logger.info(f"Processing PDF: {pdf_filename}")

    # 初始化MineruParser
    parser = MineruParser(
        lang=['ch', 'en'],  # 支持中文和英文
        parse_method='auto',  # 自动选择解析方法
        formula_enable=True,  # 启用公式解析
        table_enable=True     # 启用表格解析
    )

    try:
        # 使用MineruParser处理PDF
        result = parser.process_single_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            generate_visualizations=True,  # 生成可视化文件
            target_lang='en'  # 使用英文作为目标语言
        )

        if result["status"] == "success":
            logger.info(f"Successfully processed {pdf_filename}")
            
            # 将content.md重命名为{base_filename}.md
            original_md_path = os.path.join(output_dir, "content.md")
            new_md_path = os.path.join(output_dir, f"{base_filename}.md")
            
            if os.path.exists(original_md_path):
                import shutil
                shutil.move(original_md_path, new_md_path)
                logger.info(f"Created markdown file: {base_filename}.md")
        else:
            logger.error(f"Failed to process {pdf_filename}: {result.get('error', 'Unknown error')}")
            raise Exception(result.get('error', 'Unknown error'))

    except Exception as e:
        logger.error(f"Error processing {pdf_filename}: {str(e)}")
        raise


def main():
    """Main function to process PDFs across all topic directories."""
    logger.info("Starting PDF processing with MineruParser")
    total_pdfs = 0
    processed_pdfs = 0
    pdfs_dir = "G:/BaiduNetdiskDownload/Downloader/downloads/pdfs"
    
    if not os.path.exists(pdfs_dir):
        logger.error(f"Directory not found: {pdfs_dir}")
        return
    
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]
    total_pdfs += len(pdf_files)

    logger.info(f"Processing: ({len(pdf_files)} PDFs found)")

    for pdf_file in tqdm(pdf_files, desc="Processing"):
        base_filename = os.path.splitext(pdf_file)[0]

        # 只检查{base_filename}.md文件是否存在
        md_file = os.path.join(pdfs_dir, "output", base_filename, f"{base_filename}.md")
        
        if os.path.exists(md_file):
            print("PDF Processed Continue！")
            continue
            
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        output_dir = os.path.join(pdfs_dir, "output", base_filename)

        os.makedirs(output_dir, exist_ok=True)

        try:
            process_pdf(pdf_path, output_dir)
            processed_pdfs += 1
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")

    logger.info(f"PDF processing complete. Processed {processed_pdfs}/{total_pdfs} files.")


if __name__ == "__main__":
    main()
