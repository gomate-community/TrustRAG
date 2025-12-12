#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: app.py
@time: 2024/05/21
@contact: yanqiangmiffy@gamil.com
"""
import os
import shutil

import gradio as gr

from trustrag.applications.rag_weibo import WeiboRagApplication, ApplicationConfig
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig

# 修改成自己的配置！！！
app_config = ApplicationConfig()
app_config.llm_model_path = "H:/pretrained_models/llm/glm-4-9b-chat"
app_config.docs_path = 'H:/Projects/Weibo Insight/data/docs/weibo'
retriever_config = DenseRetrieverConfig(
    model_name_or_path="H:/pretrained_models/mteb/bge-large-zh-v1.5",
    dim=1024,
    index_path=os.path.join(PROJECT_BASE, 'output/weibo_dense')
)
rerank_config = BgeRerankerConfig(
    model_name_or_path="H:/pretrained_models/mteb/bge-reranker-large"
)

app_config.retriever_config = retriever_config
app_config.rerank_config = rerank_config
application = WeiboRagApplication(app_config)


# application.init_vector_store()


def get_file_list():
    if not os.path.exists(app_config.docs_path):
        return []
    return [f for f in os.listdir(app_config.docs_path)]


file_list = get_file_list()


def info_fn(filename):
    gr.Info(f"upload file:{filename} success!")


def upload_file(file):
    cache_base_dir = app_config.docs_path
    if not os.path.exists(cache_base_dir):
        os.mkdir(cache_base_dir)
    filename = os.path.basename(file.name)
    shutil.move(file.name, cache_base_dir + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    application.add_document(app_config.docs_path + filename)
    info_fn(filename)
    return gr.Dropdown(choices=file_list, value=filename, interactive=True)


def set_knowledge(kg_name, history):
    try:
        application.load_vector_store()
        msg_status = f'{kg_name}知识库已成功加载'
    except Exception as e:
        print(e)
        msg_status = f'{kg_name}知识库未成功加载'
    return history + [[None, msg_status]]


def clear_session():
    return '', None


def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    # print(large_language_model, embedding_model)
    print(input)
    if history == None:
        history = []

    if use_web == '使用':
        web_content = application.retriever.search_web(query=input)
    else:
        web_content = ''
    search_text = ''
    if use_pattern == '模型问答':
        result = application.get_llm_answer(query=input, web_content=web_content)
        history.append((input, result))
        search_text += web_content
        return '', history, history, search_text

    else:
        response, _, contents = application.chat(
            question=input,
            top_k=top_k,
        )
        history.append((input, response))
        for idx, source in enumerate(contents[:5]):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source}\n\n'
        # print(search_text)
        search_text += "----------【网络检索内容】-----------\n"
        search_text += web_content
        print("--------------------【模型回答】----------------\n")
        print(response)
        return '', history, history, search_text


with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""<h1><center>交互式微博舆情问答助手</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown([
                "text2vec-base",
                "bge-large-v1.5",
                "bge-base-v1.5",
            ],
                label="Embedding model",
                value="bge-large-v1.5")

            large_language_model = gr.Dropdown(
                [
                    "ChatGLM3-9B",
                ],
                label="large language model",
                value="ChatGLM3-9B")

            top_k = gr.Slider(1,
                              20,
                              value=4,
                              step=1,
                              label="检索top-k文档",
                              interactive=True)

            use_web = gr.Radio(["使用", "不使用"], label="web search",
                               info="是否使用网络搜索，使用时确保网络通常",
                               value="不使用", interactive=False
                               )
            use_pattern = gr.Radio(
                [
                    '模型问答',
                    '知识库问答',
                ],
                label="模式",
                value='知识库问答',
                interactive=False)

            kg_name = gr.Radio(["微博知识库"],
                               label="知识库",
                               value=None,
                               info="使用知识库问答，请加载知识库",
                               interactive=True)
            set_kg_btn = gr.Button("加载知识库")

            file = gr.File(label="将文件上传到知识库库，内容要尽量匹配",
                           visible=True,
                           file_types=['.txt', '.md', '.docx', '.pdf']
                           )
            # uploaded_files = gr.Dropdown(
            #     file_list,
            #     label="已上传的文件列表",
            #     value=file_list[0] if len(file_list) > 0 else '',
            #     interactive=True
            # )
        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='Weibo Insight Application',height=650)
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
            with gr.Row():
                gr.Markdown("""提醒：<br>
                                        [Weibo Insight](https://github.com/Weibo-Insight/WeiboInsight) <br>
                                        有任何使用问题[Github Issue区](https://github.com/Weibo-Insight/WeiboInsight)进行反馈. 
                                        <br>
                                        """)
        with gr.Column(scale=2):
            search = gr.Textbox(label='搜索结果')

        # ============= 触发动作=============
        file.upload(upload_file,
                    inputs=file,
                    outputs=None)
        set_kg_btn.click(
            set_knowledge,
            show_progress="full",
            inputs=[kg_name, chatbot],
            outputs=chatbot
        )
        # 发送按钮 提交
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
                   outputs=[message, chatbot, state, search])

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        # 输入框 回车
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
                       outputs=[message, chatbot, state, search])

demo.queue().launch(
    server_name='0.0.0.0',
    server_port=7860,
    share=True,
    show_error=True,
    debug=True,
    # enable_queue=True,
    inbrowser=False,
)
