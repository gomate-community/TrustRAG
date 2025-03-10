#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: app.py
@time: 2024/05/21
@contact: yanqiangmiffy@gamil.com
"""
import  sys
sys.path.append(".")
import os
import shutil
import time
import gradio as gr
import loguru
import pandas as pd

from trustrag.applications.rag_openai import RagApplication, ApplicationConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig
from datetime import datetime
import pytz
from trustrag.config.config_loader import config
# ========================== Config Start====================
# 加载配置
llm_service = config.get_config('services.dmx')
llm_model = config.get_config('models.llm')

app_config = ApplicationConfig()
app_config.docs_path = config.get_config('paths.docs')
app_config.base_url = llm_service['base_url']
app_config.api_key = llm_service['api_key']
app_config.model_name = llm_model['name']

# 加载嵌入模型配置
embedding_service = config.get_config('services.dmx')
embedding_model = config.get_config('models.embedding')

retriever_config = DenseRetrieverConfig(
    dim=3072,
    index_path=config.get_config('paths.index'),
    batch_size=32,
    api_key=embedding_service['api_key'],
    base_url=embedding_service['base_url'],
    embedding_model_name=embedding_model['name']
)
# rerank_config = BgeRerankerConfig(
#     model_name_or_path=r"H:\pretrained_models\mteb\bge-reranker-large"
# )
app_config.retriever_config = retriever_config
# app_config.rerank_config = rerank_config
application = RagApplication(app_config)
application.init_vector_store()
# ========================== Config End====================


# 创建北京时区的变量
beijing_tz = pytz.timezone("Asia/Shanghai")
IGNORE_FILE_LIST = [".DS_Store"]


class CustomUploadFile:
    def __init__(self, file):  # 修改构造函数接收 gradio 文件对象
        self.file = file  # 保存整个文件对象
        self.file_name = os.path.basename(file.name)
        self.start_time = datetime.now(beijing_tz)
        self.end_time = None
        self.duration = None
        self.state = None
        self.finished = False

    def update_process_duration(self):
        if not self.finished:
            self.end_time = datetime.now(beijing_tz)
            self.duration = (self.end_time - self.start_time).total_seconds()
            return self.duration

    def update_state(self, state):
        self.state = state

    def is_finished(self):
        self.finished = True

    def __info__(self):
        return [
            self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.duration,
            self.state,
        ]


def upload_files(
        upload_files,
        chunk_size,
        chunk_overlap,
        upload_index,
):
    if not upload_files:
        return [
            gr.update(visible=False),
            gr.update(
                visible=True,
                value="No file selected. Please choose at least one file.",
            ),
        ]
    for state_info in upload_knowledge(
            upload_files=upload_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_name=upload_index,
    ):
        yield state_info

def upload_knowledge(
        upload_files,
        chunk_size,
        chunk_overlap,
        index_name
):
    my_upload_files = []

    # 处理上传的文件
    for file in upload_files:
        if os.path.basename(file.name) not in IGNORE_FILE_LIST:
            my_upload_files.append(CustomUploadFile(file))

    result = {"Info": ["StartTime", "EndTime", "Duration(s)", "Status"]}
    error_msg = None
    success_msg = None

    while True:
        for file in my_upload_files:
            try:
                cache_base_dir = app_config.docs_path
                if not os.path.exists(cache_base_dir):
                    os.makedirs(cache_base_dir)  # 使用 makedirs 替代 mkdir

                # 使用正确的源文件路径
                source_path = file.file.name  # 使用 gradio 上传文件的临时路径
                dest_path = os.path.join(cache_base_dir, file.file_name)  # 使用 os.path.join 构建路径
                # 复制文件而不是移动，因为我们在处理临时文件
                shutil.copy2(source_path, dest_path)
                # 添加文档
                response = application.add_document(dest_path)
                file.update_state(response["status"])
                file.update_process_duration()
                result[file.file_name] = file.__info__()

                if response["status"] in ["completed", "failed"]:
                    file.is_finished()
                if response["detail"]:
                    success_msg = response["detail"]

            except Exception as api_error:
                error_msg = str(api_error)
                file.update_state("failed")
                file.is_finished()
                result[file.file_name] = file.__info__()

        yield [
            gr.update(visible=True, value=pd.DataFrame(result)),
            gr.update(visible=False),
        ]

        if all(file.finished for file in my_upload_files):
            break

        time.sleep(2)

    upload_result = f"Upload success:{success_msg}" if not error_msg else f"Upload failed: {error_msg}"
    yield [
        gr.update(visible=True, value=pd.DataFrame(result)),
        gr.update(visible=True, value=upload_result),
    ]

def clear_files():
    yield [
        gr.update(visible=False, value=pd.DataFrame()),
        gr.update(visible=False, value=""),
    ]


def get_file_info(file_path):
    """Get detailed information about a file"""
    stats = os.stat(file_path)
    creation_time = datetime.fromtimestamp(stats.st_ctime, beijing_tz)
    modified_time = datetime.fromtimestamp(stats.st_mtime, beijing_tz)

    return {
        "Filename": os.path.basename(file_path),
        "Size(KB)": round(stats.st_size / 1024, 2),
        "CreateTime": creation_time.strftime("%Y-%m-%d %H:%M:%S"),
        "UpdateTime": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
        "FileType": os.path.splitext(file_path)[1].lower()
    }


def list_documents():
    """List all documents in the docs directory"""
    if not os.path.exists(app_config.docs_path):
        return pd.DataFrame()

    files_info = []
    for file_name in os.listdir(app_config.docs_path):
        if file_name not in IGNORE_FILE_LIST:
            file_path = os.path.join(app_config.docs_path, file_name)
            if os.path.isfile(file_path):
                files_info.append(get_file_info(file_path))

    return pd.DataFrame(files_info)

def refresh_file_list():
    return gr.update(value=list_documents())



def delete_selected_file(selected_file_name):
    """Delete the selected file and return updated file list"""
    if not selected_file_name:
        gr.Info("Please select a file first")
        return gr.update(value=list_documents()), gr.update(visible=False)

    try:
        file_path = os.path.join(app_config.docs_path, selected_file_name)
        if os.path.exists(file_path):
            gr.Info(f"Successfully deleted file and rebuild faiss index: {selected_file_name}")
            os.remove(file_path)
            application.init_vector_store()
            return gr.update(value=list_documents()), gr.update(visible=False)
        else:
            gr.Warning(f"File not found: {selected_file_name}")
            return gr.update(value=list_documents()), gr.update(visible=False)
    except Exception as e:
        gr.Error(f"Error deleting file: {str(e)}")
        return gr.update(value=list_documents()), gr.update(visible=False)


def get_file_chunks(file_path, chunk_size):
    """Get chunks from a file and return as a DataFrame"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = application.tc.get_chunks([content],chunk_size)

        # Create DataFrame with chunk information
        chunks_data = []
        for idx, chunk in enumerate(chunks, 1):
            chunks_data.append({
                "Chunk ID": idx,
                "Content": chunk,
                "Length": len(chunk)
            })

        return pd.DataFrame(chunks_data)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return pd.DataFrame()


def on_file_select(files_df, chunk_size, evt: gr.SelectData):
    """Handle file selection event"""
    if not evt.value:
        return None, None, gr.update(visible=False), pd.DataFrame()

    try:
        selected_filename = files_df.to_dict('records')[evt.index[0]]['Filename']
        file_path = os.path.join(app_config.docs_path, selected_filename)

        # Get chunks information
        chunks_df = get_file_chunks(file_path, chunk_size)

        return selected_filename, f"Selected file: {selected_filename}", gr.update(visible=True), gr.update(visible=True,value=chunks_df)
    except (KeyError, IndexError):
        return None, None, gr.update(visible=False), pd.DataFrame()

def clear_session():
    return '', None


def shorten_label(text, max_length=10):
    if len(text) > 2 * max_length:
        return text[:max_length] + "..." + text[-max_length:]
    return text


def predict(question,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    loguru.logger.info("User Question：" + question)
    if history is None:
        history = []
    # Handle web content
    web_content = ''
    if use_web == 'Use':
        loguru.logger.info("Use Web Search")
        results = application.web_searcher.retrieve(query=question, top_k=5)
        for search_result in results:
            web_content += search_result['title'] + " " + search_result['body'] + "\n"
    search_text = ''
    if use_pattern == 'Only LLM':
        # Handle model Q&A mode
        loguru.logger.info('Only LLM Mode:')

        # result = application.llm.chat(query=question, web_content=web_content)
        system_prompt = "You are a helpful assistant."
        user_input = [
            {"role": "user", "content": question}
        ]
        # 调用 chat 方法进行对话
        result, total_tokens = application.llm.chat(system=system_prompt, history=user_input)
        history.append((question, result))
        search_text += web_content

        # Return empty judge results for Q&A mode
        checkboxes = []
        for item in range(5):
            checkbox = gr.Checkbox(value=False, visible=False, interactive=False)
            checkboxes.append(checkbox)
        return '', history, history, search_text, '', checkboxes[0], checkboxes[1], checkboxes[2], checkboxes[3], \
            checkboxes[4]

    else:
        # Handle RAG mode
        loguru.logger.info('RAG Mode:')
        response, _, contents, rewrite_query = application.chat(
            question=question,
            top_k=top_k,
        )
        history.append((question, response))
        # Format search results
        for idx, source in enumerate(contents):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source["text"]}\n分数：{source["score"]:.2f}\n\n'
        # Add web content if available
        if web_content:
            search_text += "----------【网络检索内容】-----------\n"
            search_text += web_content
        checkboxes = []
        for idx,item in enumerate(contents[:5]):
            checked = bool(item.get('label', 0))
            label_text = item.get('text', '')
            shortened_label = str(idx+1)+"."+shorten_label(label_text)
            checkbox = gr.Checkbox(value=checked, visible=True, label=shortened_label, interactive=True)
            checkboxes.append(checkbox)
        return '', history, history, search_text, rewrite_query, checkboxes[0], checkboxes[1], checkboxes[2], \
            checkboxes[3], checkboxes[4]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""<h1><center>TrustRAG Studio</center></h1><center><font size=3></center></font>""")
    # ===== tab start
    with gr.Tab("\N{rocket} Corpus"):
        with gr.Row():
            with gr.Column(scale=2):
                upload_index = gr.Dropdown(
                    choices=["default_index"],
                    value="default_index",
                    label="\N{book} Knowledge Name",
                    elem_id="knowledge_name",
                )
                chunk_size = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=128,
                    step=64,
                    label="\N{GEAR} Chunk Size",
                    info="Split Document With The Chunk Size",
                    interactive=True
                )

                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=128,
                    value=0,
                    step=1,
                    label="\N{GEAR} Chunk Overlap",
                    info="Chunk Overlap Within Chunks",
                    interactive=True
                )

                enable_decontextualization = gr.Checkbox(
                    label="Yes",
                    value=True,
                    info="Process with Contextual Decontextualization",
                    elem_id="enable_Decontextualization",
                    visible=True,
                )
            with gr.Column(scale=8):
                selected_file = gr.State(None)

                files_df = gr.DataFrame(
                    value=list_documents(),
                    label="Current Documents",
                    interactive=False,
                )
                selected_file_info = gr.Markdown(visible=True)
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Files List")
                    delete_btn = gr.Button("Delete Selected File", visible=False)  # Initially hidden delete button
                with gr.Row():
                    # Add chunks display DataFrame
                    chunks_display = gr.DataFrame(
                            label="Document Chunks",
                            value=pd.DataFrame(),
                            visible=False,
                            interactive=False
                    )

                refresh_btn.click(fn=refresh_file_list, outputs=[files_df])
                files_df.select(
                    fn=on_file_select,
                    inputs=[files_df, chunk_size],
                    outputs=[selected_file, selected_file_info, delete_btn, chunks_display]
                )

                # Handle file deletion
                delete_btn.click(
                    fn=delete_selected_file,
                    inputs=[selected_file],
                    outputs=[files_df, delete_btn]
                )

                with gr.Tab("Files"):
                    upload_file = gr.File(
                        label="Upload a knowledge file.", file_count="multiple"
                    )
                    upload_file_state_df = gr.DataFrame(
                        label="Upload Status Info", visible=False
                    )
                    upload_file_state = gr.Textbox(label="Upload Status", visible=False)
                with gr.Tab("Directory"):
                    upload_file_dir = gr.File(
                        label="Upload a knowledge directory.",
                        file_count="directory",
                    )
                    upload_dir_state_df = gr.DataFrame(
                        label="Upload Status Info", visible=False
                    )
                    upload_dir_state = gr.Textbox(label="Upload Status", visible=False)

                upload_file.upload(
                    fn=upload_files,
                    inputs=[
                        upload_file,
                        chunk_size,
                        chunk_overlap,
                        upload_index,
                    ],
                    outputs=[upload_file_state_df, upload_file_state],
                    api_name="upload_knowledge",
                )
                upload_file.clear(
                    fn=clear_files,
                    inputs=[],
                    outputs=[upload_file_state_df, upload_file_state],
                    api_name="clear_file",
                )
                dummy_component = gr.Textbox(visible=False, value="")
                upload_file_dir.upload(
                    fn=upload_knowledge,
                    inputs=[
                        upload_file_dir,
                        chunk_size,
                        chunk_overlap,
                        upload_index,
                    ],
                    outputs=[upload_dir_state_df, upload_dir_state],
                    api_name="upload_knowledge_dir",
                )
                upload_file_dir.clear(
                    fn=clear_files,
                    inputs=[],
                    outputs=[upload_dir_state_df, upload_dir_state],
                    api_name="clear_file_dir",
                )
    with gr.Tab("\N{fire} Chat"):
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                embedding_model = gr.Dropdown(
                    choices=[
                        "text-embedding-3-large"
                    ],
                    label="Embedding model",
                    value="text-embedding-3-large"
                )

                large_language_model = gr.Dropdown(
                    choices=[
                        "GPT-4O-ALL",
                    ],
                    label="Large Language model",
                    value="GPT-4O-ALL"
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Retrieve Top-k Documents",
                    interactive=True
                )

                use_web = gr.Radio(
                    choices=["Use", "Not used"],
                    label="Web Search",
                    info="Do you use network search? When using it, make sure the network is normal.",
                    value="Not used",
                    interactive=True
                )
                use_pattern = gr.Radio(
                    choices=[
                        'Only LLM',
                        'RAG',
                    ],
                    label="Chat Mode",
                    value='RAG',
                    interactive=True
                )
                knowledge_name = gr.Dropdown(
                    choices=[
                        "default_index",
                    ],
                    label="Knowledge Name",
                    value="default_index",
                    interactive=True
                )
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(label='TrustRAG Application', height=650)
                with gr.Row():
                    message = gr.Textbox(label='Please enter a question')
                with gr.Row():
                    clear_history = gr.Button(" Clear")
                    send = gr.Button(" Send")
                with gr.Row():
                    gr.Markdown(
                        """>Remind：[TrustRAG Application](https://github.com/gomate-community/TrustRAG/issues)If you have any questions, please provide feedback in [Github Issue区](https://github.com/gomate-community/TrustRAG/issues) .""")
            with gr.Column(scale=2):
                with gr.Row():
                    rewrite = gr.Textbox(label='Query Reformulate')
                with gr.Row():
                    # todo:创建judge显示结果，使用复选框
                    with gr.Column() as checkbox_container:
                        # gr.Markdown("Document Judge")
                        checkbox_outputs = [gr.Checkbox(visible=False, interactive=True) for _ in range(5)]
                with gr.Row():
                    search = gr.Textbox(label='Claim Attribute')

            # submit
            send.click(predict,
                       inputs=[
                           message,
                           large_language_model,
                           embedding_model,
                           top_k,
                           use_web,
                           use_pattern,
                           state
                       ],
                       outputs=[message, chatbot, state, search, rewrite] + checkbox_outputs)

            # clear
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)
            # enter
            message.submit(predict,
                           inputs=[
                               message,
                               large_language_model,
                               embedding_model,
                               top_k,
                               use_web,
                               use_pattern,
                               state
                           ],
                           outputs=[message, chatbot, state, search, rewrite] + checkbox_outputs)
    with gr.Tab("\N{book} DeepRsearch"):
        with gr.Row():
            gr.Markdown(
                """>Remind：[TrustRAG Application](https://github.com/gomate-community/TrustRAG/issues)If you have any questions, please provide feedback in [Github Issue区](https://github.com/gomate-community/TrustRAG/issues) .""")

demo.queue(max_size=2).launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=True,
    show_error=True,
    debug=True,
    inbrowser=False,
)
