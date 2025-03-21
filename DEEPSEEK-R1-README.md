# DeepSeek-R1 模型集成

本分支添加了对 SiliconFlow API 的支持，使 TrustRAG 框架能够使用 DeepSeek-R1 模型进行检索增强生成和深度研究。

## 主要特性

1. **SiliconFlow API 集成**：
   - 添加了 SiliconFlow API 端点配置
   - 支持 DeepSeek-R1 等一系列高性能模型

2. **DeepResearch 模块增强**：
   - 改进了响应解析机制，支持 reasoning_content 字段
   - 优化了异常处理，提高了系统稳定性
   - 添加了结构化数据转换，处理不同格式的响应

3. **Web 应用支持**：
   - 在应用界面添加了 DeepSeek-R1 模型选项
   - 实现了根据选择的模型动态切换 API 服务
   - 维护了统一的用户体验

## 如何使用

### 配置 SiliconFlow API

在 `.env` 文件（或 `config_online.json`）中添加以下配置：

```bash
# SiliconFlow (DeepSeek-R1)
SILICONFLOW_API_KEY="your_api_key_here"
SILICONFLOW_MODEL="deepseek-ai/DeepSeek-R1"
SILICONFLOW_ENDPOINT="https://api.siliconflow.cn/v1"
```

### 运行 DeepResearch

```bash
cd trustrag/modules/deepresearch
python pipeline.py
```

在提示时选择研究主题，系统将使用 DeepSeek-R1 模型进行深度研究并生成详细报告。

### 使用 Web 界面

```bash
python app.py
```

在 Web 界面中选择 "DeepSeek-R1" 模型进行问答。

## 支持的模型

SiliconFlow API 支持多种强大的模型，包括：

- deepseek-ai/DeepSeek-R1 (默认)
- deepseek-ai/DeepSeek-V3
- Qwen/QwQ-32B
- 更多模型请参考 SiliconFlow 文档

## 技术详情

本集成通过 OpenAI 兼容 API 接口调用 SiliconFlow 服务，并对 DeepSeek-R1 模型的特殊响应格式（如 reasoning_content 字段）进行了专门处理，确保了系统能够充分利用模型的推理能力。 