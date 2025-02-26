## 安装环境

下面是建议安装步骤
- 第一步：设置pip清华源

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

- 第二步：安装torch环境

> https://pytorch.org/get-started/locally/

安装命令如下：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- 第三步：安装其他依赖
```bash
pip install -r  requirements.txt
```

安装成功之后，显示信息如下：
```text
Successfully installed FlagEmbedding-1.3.4 PyMuPDF-1.25.3 PyPDF2-3.0.1 XlsxWriter-3.2.2 accelerate-1.4.0 aiofiles-23.2.1 aiosqlite-0.20.0 asgiref-3.8.1 backoff-2.2.1 bcrypt-4.2.1 bm25s-0.2.7.post1 bson-0.5.10 build-1.2.2.post1 cbor-1.0.0 chroma-hnswlib-0.7.6 chromadb-0.6.3 click-8.1.8 codecov-2.1.13 colored
logs-15.0.1 coverage-7.6.12 dashscope-1.22.1 dataclasses-json-0.6.7 datasets-3.3.2 datrie-0.8.2 deprecated-1.2.18 dnspython-2.7.0 duckduckgo-search-7.5.0 durationpy-0.9 elastic-transport-8.17.0 elasticsearch-8.17.1 faiss-cpu-1.10.0 fastapi-0.115.8 fastembed-0.5.1 ffmpy-0.5.0 flake8_docstrings-1.7.0 flatbuff
ers-25.2.10 future-1.0.0 google-auth-2.38.0 googleapis-common-protos-1.68.0 gradio-3.50.2 gradio-client-0.6.1 grpcio-1.67.1 grpcio-tools-1.67.1 h2-4.2.0 hanziconv-0.3.2 hpack-4.1.0 html2text-2024.2.26 html_text-0.7.0 httptools-0.6.4 httpx-sse-0.4.0 huggingface-hub-0.29.1 humanfriendly-10.0 hyperframe-6.1.0 
hyperopt-0.2.7 ijson-3.3.0 importlib-resources-6.5.2 inscriptis-2.5.3 ir-datasets-0.5.9 iso8601-2.1.0 jieba-0.42.1 jiter-0.8.2 kubernetes-32.0.1 langchain-0.3.19 langchain-community-0.3.18 langchain-core-0.3.39 langchain-huggingface-0.1.2 langchain-openai-0.3.7 langchain-text-splitters-0.3.6 langsmith-0.3.1
1 loguru-0.7.3 lxml-5.3.1 lxml-html-clean-0.4.1 marshmallow-3.26.1 minio-7.2.15 mmh3-4.1.0 monotonic-1.6 multiprocess-0.70.16 oauthlib-3.2.2 onnxruntime-1.20.1 openai-1.64.0 opentelemetry-api-1.30.0 opentelemetry-exporter-otlp-proto-common-1.30.0 opentelemetry-exporter-otlp-proto-grpc-1.30.0 opentelemetry-i
nstrumentation-0.51b0 opentelemetry-instrumentation-asgi-0.51b0 opentelemetry-instrumentation-fastapi-0.51b0 opentelemetry-proto-1.30.0 opentelemetry-sdk-1.30.0 opentelemetry-semantic-conventions-0.51b0 opentelemetry-util-http-0.51b0 orjson-3.10.15 pdfminer.six-20231228 pdfplumber-0.11.5 peft-0.14.0 portalocker-2.10.1 posthog-3.15.1 primp-0.14.0 protobuf-5.29.3 py-rust-stemmers-0.1.5 py4j-0.10.9.9 pycryptodome-3.21.0 pydantic-settings-2.8.0 pydub-0.25.1 pymilvus-2.5.4 pymongo-4.8.0 pynndescent-0.5.13 pypdfium2-4.30.1 pypika-0.48.9 pypika-tortoise-0.5.0 pyproject_hooks-1.2.0 pyreadline3-3.5.4 pytest-cov-6.0.0 python-docx-1.1.2 python-dotenv-1.0.1 python-magic-0.4.27 python-multipart-0.0.20 python-pptx-1.0.2 qdrant-client-1.13.2 readability-0.3.2 requests-oauthlib-2.0.0 rsa-4.9 safetensors-0.5.2 semantic-version-2.10.0 sentence_transformers-3.4.1 sentencepiece-0.2.0 shellingham-1.5.4 starlette-0.45.3 tiktoken-0.9.0 tokenizers-0.20.3 tortoise-orm-0.24.1 transformers-4.46.0 trec-car-tools-2.6 typer-0.15.1 typing-inspect-0.9.0 umap-learn-0.5.7 unlzw3-0.2.3 uvicorn-0.34.0 warc3-wet-0.2.5 warc3-wet-clueweb09-0.2.5 watchfiles-1.0.4 websockets-11.0.3 win32-setctime-1.2.0 xgboost-2.1.4 xpinyin-0.7.6 xxhash-3.5.0 zhipuai-2.1.5.20250106 zlib-state-0.1.9
```

- 第四步：下载nltk所需包
如果直接运行app.py可能直接报错如下：

```text
  Resource punkt_tab not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('punkt_tab')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load tokenizers/punkt_tab/english/

  Searched in:
    - 'C:\\Users\\yanqiang/nltk_data'
    - 'F:\\ProgramData\\anaconda3\\nltk_data'
    - 'F:\\ProgramData\\anaconda3\\share\\nltk_data'
    - 'F:\\ProgramData\\anaconda3\\lib\\nltk_data'
    - 'C:\\Users\\yanqiang\\AppData\\Roaming\\nltk_data'
    - 'C:\\nltk_data'
    - 'D:\\nltk_data'
    - 'E:\\nltk_data'
**********************************************************************
```
大家可以按照上面提示进行下载，如果网络不好的话，下面是上传的百度链接，大家直接解压之后，放在上面任意目录就行，下载地址如下
```text
通过百度网盘分享的文件：nltk_data.zip
链接：https://pan.baidu.com/s/1UPCtbX6lCuKN-7T_9fWQNA?pwd=ambt 
提取码：ambt 
--来自百度网盘超级会员V5的分享
```

视频教程已上传到B站

**TrustRAG框架使用基础教程**

https://www.bilibili.com/video/BV1yePve3Ebi/

## FAQ

- datrie安装失败

    windows可能安装datrie失败，具体错误如下：
    ```text
    
     error: subprocess-exited-with-error
    
      × Building wheel for datrie (pyproject.toml) did not run successfully.
      │ exit code: 1
      ╰─> [7 lines of output]
          C:\Users\yanqiang\AppData\Local\Temp\pip-build-env-mo4qgjqp\overlay\Lib\site-packages\setuptools\_distutils\dist.py:270: UserWarning: Unknown distribution option: 'tests_require'
            warnings.warn(msg)
          running bdist_wheel
          running build
          running build_clib
          building 'datrie' library
          error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
          [end of output]
    
      note: This error originates from a subprocess, and is likely not a problem with pip.
      ERROR: Failed building wheel for datrie
    
    ```
    
    解决方法：下载生成工具：https://visualstudio.microsoft.com/visual-cpp-build-tools/ 。 下载完成后，直接运行，看到如下画面，选择C++桌面开发，右侧选择默认即可

    参考资料：https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/133496352