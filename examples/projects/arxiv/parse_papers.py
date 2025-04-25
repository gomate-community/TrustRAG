import os
import logging
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from tqdm import tqdm
import torch
print(torch.cuda.is_available())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pdf_processor')

# Target directories to process
TOPIC_DIRECTORIES = [
    # "papers/topic_Chain_of_Thought",
    # "papers/topic_LLM_Post-Training",
    # "papers/topic_Reasoning_Large_Language_Models",
    # "papers/topic_Retrieval_Augmented_Generation_RAG",
    # "papers/topic_Continual_Learning",
    "papers/topic_Multi-Agent",
    "papers/topic_Tool_Learning",
    "papers/topic_Multi-Step_Reasoning",
    "papers/topic_Fine-Tuning",
    "topic_LLM_Based_Agent",
    "topic_In-Context_Learning",
    "topic_RLHF",
    "topic_Pre-Training"
]


def process_pdf(pdf_path, output_dir):
    """
    Process a PDF file and generate various output files.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory where outputs will be saved
    """
    pdf_filename = os.path.basename(pdf_path)
    base_filename = os.path.splitext(pdf_filename)[0]

    logger.info(f"Processing PDF: {pdf_filename}")

    # Prepare directory structure
    images_dir_path = os.path.join(output_dir, "images")
    images_dir_name = os.path.basename(images_dir_path)

    os.makedirs(images_dir_path, exist_ok=True)
    logger.debug(f"Created images directory: {images_dir_path}")

    # Initialize file writers
    image_writer = FileBasedDataWriter(images_dir_path)
    md_writer = FileBasedDataWriter(output_dir)

    # Read PDF content
    pdf_reader = FileBasedDataReader("")
    pdf_bytes = pdf_reader.read(pdf_path)
    logger.debug(f"Read {len(pdf_bytes)} bytes from {pdf_filename}")

    # Process PDF
    dataset = PymuDocDataset(pdf_bytes)
    pdf_type = dataset.classify()
    logger.info(f"Detected PDF type: {pdf_type}")

    # Apply appropriate processing based on PDF type
    if pdf_type == SupportedPdfParseMethod.OCR:
        logger.info(f"Using OCR mode for {pdf_filename}")
        inference_result = dataset.apply(doc_analyze, ocr=True)
        processing_result = inference_result.pipe_ocr_mode(image_writer)
    else:
        logger.info(f"Using text mode for {pdf_filename}")
        inference_result = dataset.apply(doc_analyze, ocr=False)
        processing_result = inference_result.pipe_txt_mode(image_writer)

    # Generate output files
    logger.debug("Generating output files")
    model_pdf_path = os.path.join(output_dir, "model.pdf")
    inference_result.draw_model(model_pdf_path)
    logger.debug(f"Created model visualization: {model_pdf_path}")

    model_inference_result = inference_result.get_infer_res()

    layout_pdf_path = os.path.join(output_dir, "layout.pdf")
    processing_result.draw_layout(layout_pdf_path)
    logger.debug(f"Created layout visualization: {layout_pdf_path}")

    spans_pdf_path = os.path.join(output_dir, "spans.pdf")
    processing_result.draw_span(spans_pdf_path)
    logger.debug(f"Created spans visualization: {spans_pdf_path}")

    # Generate markdown content
    markdown_content = processing_result.get_markdown(images_dir_name)
    markdown_path = f"{base_filename}.md"
    processing_result.dump_md(md_writer, markdown_path, images_dir_name)
    logger.info(f"Created markdown file: {markdown_path}")

    # Generate content list
    content_list = processing_result.get_content_list(images_dir_name)
    processing_result.dump_content_list(md_writer, "content_list.json", images_dir_name)
    logger.debug("Created content list JSON")

    # Generate middle JSON
    middle_json = processing_result.get_middle_json()
    processing_result.dump_middle_json(md_writer, "middle.json")
    logger.debug("Created middle JSON file")

    logger.info(f"Successfully processed {pdf_filename}")


def main():
    """Main function to process PDFs across all topic directories."""
    logger.info("Starting PDF processing")
    total_pdfs = 0
    processed_pdfs = 0

    for topic_dir in TOPIC_DIRECTORIES:
        pdfs_dir = os.path.join(topic_dir, "pdfs")
        if not os.path.exists(pdfs_dir):
            logger.warning(f"Directory not found: {pdfs_dir}")
            continue

        pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")]
        total_pdfs += len(pdf_files)

        logger.info(f"Processing topic: {topic_dir} ({len(pdf_files)} PDFs found)")

        for pdf_file in tqdm(pdf_files, desc=f"Processing {os.path.basename(topic_dir)}"):
            base_filename = os.path.splitext(pdf_file)[0]

            md_file=os.path.join(topic_dir, "output", base_filename, f"{base_filename}.md")
            if os.path.exists(md_file):
                print("PDF Processed Continue！")
                continue
            pdf_path = os.path.join(pdfs_dir, pdf_file)
            output_dir = os.path.join(topic_dir, "output", base_filename)

            os.makedirs(output_dir, exist_ok=True)

            try:
                process_pdf(pdf_path, output_dir)
                processed_pdfs += 1
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")

    logger.info(f"PDF processing complete. Processed {processed_pdfs}/{total_pdfs} files.")


if __name__ == "__main__":
    main()