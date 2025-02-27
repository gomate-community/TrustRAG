>本文档旨在为用户提供关于 DeepSeek-R1-Distill-Qwen-32B 模型服务的调用及相关接口的详细说明。备注：有技术相关问题请联系`yanqiang`。

## 1. 概述

本文档旨在为用户提供关于 **DeepSeek-R1-Distill-Qwen** 模型服务的调用及相关接口的详细说明。该服务基于 [vLLM](https://github.com/vllm-project/vllm) 实现，并兼容 OpenAI API 协议，支持文本生成（Completions）和对话生成（Chat Completions）任务。

### 部署命令

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
  --model /data/users/searchgpt/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --served-model-name DeepSeek-R1-Distill-Qwen-32B \
  --max-model-len=8192 \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size=4 \
  --host 0.0.0.0 \
  --port 8000 \
```

### 服务地址
- **文档地址**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **服务地址**: `http://127.0.0.1:8000`

---

## 2. 接口说明

### 2.1 获取模型列表

通过以下命令获取当前托管的模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

#### 返回示例：
```json
{
  "object": "list",
  "data": [
    {
      "id": "DeepSeek-R1-Distill-Qwen-32B",
      "object": "model",
      "created": 1739935000,
      "owned_by": "vllm",
      "root": "DeepSeek-R1-Distill-Qwen-32B",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-d076145fbcad40c1837cbe2ecb281fd2",
          "object": "model_permission",
          "created": 1739935000,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

---

### 2.2 文本生成（Completions）

#### 请求方式
使用 `curl` 命令或 Python 脚本调用 `/v1/completions` 接口。

##### curl 示例：
```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-R1-Distill-Qwen-32B",
        "prompt": "请介绍下中科院计算所？",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

##### 返回示例：
```json
{
  "id": "cmpl-cd1a9c5b29c84e549e2259869aeadd3d",
  "object": "text_completion",
  "created": 1739935318,
  "model": "DeepSeek-R1-Distill-Qwen-32B",
  "choices": [
    {
      "index": 0,
      "text": " 中科院计算所，全称中国科学院计算技术研究所，成立于1958年，是中国计算机科学领域的顶尖研究机构。它在计算机体系结构、人工智能、高性能计算、计算机网络、信息安全等领域有着显著的研究成果。计算所不仅培养了大量计算机领域的顶尖人才，还孵化了许多重要的科技企业，如联想集团。计算所的研究成果在推动中国信息技术发展方面起到了关键作用。\n\n此外，计算所还承担了多项国家重大科技项目，如“龙芯”处理器的研发，为中国在芯片设计领域提供了重要支持。计算所的科研人员在国际顶级学术会议和期刊上发表了大量高水平论文，提升了中国在国际计算机科学领域的影响力。",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 152,
    "completion_tokens": 144
  }
}
```

##### Python 示例：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {"role": "user", "content": "请介绍下中科院计算所"}
    ]
)

print(completion.choices[0].message)
```

输出如下：

```python
ChatCompletionMessage(content='好的，我现在要研究一下中国科学院计算技术研究所（简称中科院计算所）。首先，我需要了解这个机构的基本情况。从名字上看，它属于中国科学院，专注于计算技术，可能包括计算机科学和相关领域的研究。\n\n接下来，我想知道它的成立时间。根据之前的资料，它成立于1956年，是中科院首批成立的研究所之一。这个时间点很重要，因为它是在新中国成立初期，国家开始重视科技发展的时候建立的。\n\n然后，我应该了解它的研究领域。计算所涵盖计算机体系结构、操作系统、编译器、计算机网络、人工智能、大数据、计算机安全等多个方向。这说明它在计算机科学的多个关键领域都有深入研究。\n\n在重要成就方面，计算所参与了多项国家重大科技项目，比如“863”计划和“973”计划。这表明该所在国家科技发展中扮演了重要角色。此外，他们开发了龙芯系列高性能通用处理器，这在中国芯片自主研发中具有重要意义。还有并行计算机系统曙光系列，这在超级计算领域有贡献。另外，中文信息处理方面的技术也提升了国家在信息技术领域的竞争力。\n\n关于科研平台和设施，计算所拥有多个重点实验室，如计算机体系结构国家重点实验室、智能信息处理重点实验室等。这些实验室为他们的研究提供了坚实的基础。同时，他们还参与了国家超级计算中心的建设，说明他们在高性能计算方面有实力。\n\n人才培养方面，计算所应该培养了大量计算机领域的顶尖人才。他们与中国顶尖高校如清华、北大合作，应该有联合培养项目，提升教育和研究水平。\n\n国际合作方面，计算所可能与国际知名的研究机构和高校合作，促进技术交流，提升国际影响力。这可能包括联合研究项目、学术交流等。\n\n最后，社会贡献方面，计算所的研究成果转化应用，推动了信息产业的发展，提升了国家信息安全和经济竞争力。特别是在支持国家重大工程和信息安全方面，他们的贡献不可忽视。\n\n总结一下，中科院计算所作为中国计算机领域的顶尖研究机构，在技术研发、人才培养、国际合作等方面都有显著成就，对国家科技发展有重要推动作用。\n</think>\n\n中国科学院计算技术研究所（简称中科院计算所）是中国计算机科学领域的顶尖研究机构，成立于1956年，是新中国首批成立的科研院所之一。该所致力于计算机体系结构、人工智能、大数据、信息安全等领域的研究，参与了多项国家级科技项目，如“863”和“973”计划。其重要成果包括龙芯处理器和曙光超级计算机，提升了中国在信息技术领域的自主创新能力。计算所拥有多个国家级实验室，与国内外顶尖高校合作，培养了大量人才，并在国际舞台上享有盛誉，为国家的科技进步和社会发展做出了重要贡献。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None)
```

---

### 2.3 对话生成（Chat Completions）

#### 请求方式
使用 `curl` 命令或 Python 脚本调用 `/v1/chat/completions` 接口。

##### curl 示例：
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-R1-Distill-Qwen-32B",
        "messages": [
            {"role": "user", "content": "我想问你，5的阶乘是多少？<think>\n"}
        ]
    }'
```

##### 返回示例：
```json
{
  "id": "cmpl-000a3686f69d4ad18088d8816548462b",
  "object": "chat.completion",
  "created": 1739935616,
  "model": "DeepSeek-R1-Distill-Qwen-32B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "首先，阶乘是指一个正整数n，乘以比它小的所有正整数，直到1的乘积。因此，7的阶乘就是7乘以6、5、4、3、2和1。\n\n计算的步骤如下：\n7 × 6 = 42\n42 × 5 = 210\n210 × 4 = 840\n840 × 3 = 2520\n2520 × 2 = 5040\n5040 × 1 = 5040\n\n因此，7的阶乘等于5040。\n</think>\n\n要计算7的阶乘（记作7!），我们需要将7乘以比它小的所有正整数，直到1。具体计算步骤如下：\n\n\\[\n7! = 7 \\times 6 \\times 5 \\times 4 \\times 3 \\times 2 \\times 1\n\\]\n\n让我们逐步进行计算：\n\n1. \\(7 \\times 6 = 42\\)\n2. \\(42 \\times 5 = 210\\)\n3. \\(210 \\times 4 = 840\\)\n4. \\(840 \\times 3 = 2520\\)\n5. \\(2520 \\times 2 = 5040\\)\n6. \\(5040 \\times 1 = 5040\\)\n\n所以，7的阶乘等于：\n\n\\[\n7! = \\boxed{5040}\n\\]"
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "total_tokens": 358,
    "completion_tokens": 340
  }
}
```

##### Python 示例：
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
    base_url="http://127.0.0.1:8000/v1",
)

chat_outputs = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {"role": "user", "content": ""},
    ]
)

print(chat_outputs)
```

示例输出

```python
ChatCompletion(id='cmpl-47f43ac504644812b9b982f440bcb1a3', choices=[
Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='嗯，我现在要弄清楚什么是信息检索。可能我之前听说过这个词，但具体是什么意思呢？让我慢慢思考一下。\n\n信息检索，听起来像是寻找信息的过程。那它具体指的是什么呢？是不是像在电脑上搜索文件或者在图书馆找书一样？或者说是在互联网上用搜索引擎找东西？比如用谷歌搜索某个关键词，然后得到一堆结果，这就是信息检索的一部分吗？\n\n那信息检索的定义可能是什么呢？根据之前查到的内容，信息检索是指从大量信息中快速、准确地找到所需信息的过程。这可能涉及到计算机科学中的技术，比如搜索引擎、数据库查询等。那它是不是不仅仅指互联网搜索，还包括其他地方的信息查找，比如公司内部的文档管理系统或者学术数据库？\n\n信息检索涉及哪些核心技术呢？我记得提到过索引、排名和相关性算法。那索引是什么？是不是把大量文档中的关键词提取出来，建立一个索引库，方便快速查找？比如在数据库中，索引可以加快查询速度。排名算法，是不是像谷歌的PageRank那样，根据网页的重要性来排序结果？相关性算法可能是指计算查询词与文档内容的相关程度，从而决定显示哪些结果。\n\n信息检索的应用领域有哪些？搜索引擎肯定是其中之一，比如谷歌、百度这些。除此之外，还有学术数据库，比如PubMed或者IEEE Xplore，用户可以在里面查找相关的研究论文。企业内部的文档管理系统，比如公司员工可以通过内部系统查找之前的工作文档或者报告。智能助手，比如Siri或者Alexa，当用户提问时，它们可能也在进行某种形式的信息检索，从知识库中找到最合适的回答。\n\n信息检索的发展趋势是什么呢？我记得提到了大数据、人工智能、自然语言处理和个性化推荐。那大数据时代，信息量爆炸，如何高效检索信息变得越来越重要。人工智能可能在这里发挥了作用，比如机器学习算法可以改进检索的准确性。自然语言处理，可能让检索更智能化，比如理解用户的查询意图，而不是仅仅匹配关键词。个性化推荐，可能根据用户的搜索历史和偏好，提供更相关的结果，比如亚马逊根据用户的购买记录推荐商品。\n\n不过，我对这些概念还不是很清楚，比如索引具体是怎么建立的？排名算法又是如何工作的？相关性算法有什么具体的例子吗？还有大数据和人工智能在信息检索中的具体应用是怎样的？这些可能需要进一步学习和理解。\n\n另外，信息检索和数据挖掘有什么区别？我记得数据挖掘是从大量数据中发现模式和知识，而信息检索主要是找到特定的信息。它们可能有重叠的地方，但侧重点不同。\n\n有没有实际的例子可以帮助我更好地理解信息检索？比如，当我用谷歌搜索“人工智能”时，谷歌是如何快速找到相关网页并排序的？是不是先通过索引找到所有包含“人工智能”的网页，然后用PageRank来排序，再根据相关性算法调整结果，最后展示出来？\n\n还有，信息检索在实际应用中面临哪些挑战？比如如何处理大量的信息，如何提高准确性，如何应对用户的多样化需求，或者如何防止信息过载？\n\n总的来说，信息检索是一个涉及多个技术领域的复杂过程，包括计算机科学、信息学、人工智能等。要深入理解它，可能需要学习这些相关领域的知识，并了解最新的技术和应用情况。\n</think>\n\n信息检索是一个涉及从大量信息中快速、准确地找到所需信息的过程，广泛应用于搜索引擎、学术数据库、企业文档管理和智能助手等领域。它结合了多种核心技术，如索引、排名和相关性算法，并正朝着大数据、人工智能、自然语言处理和个性化推荐方向发展。以下是对信息检索的详细解析：\n\n1. **定义与范围**：\n   - 信息检索不仅限于互联网搜索，还包括图书馆、企业内部系统和学术数据库等场景。\n   - 涉及计算机科学中的技术，如搜索引擎和数据库查询。\n\n2. **核心技术**：\n   - **索引**：通过提取关键词建立索引库，加快查找速度。例如，数据库索引提高查询效率。\n   - **排名算法**：如谷歌的PageRank，根据网页重要性排序结果。\n   - **相关性算法**：评估查询词与文档的相关性，决定结果展示。\n\n3. **应用领域**：\n   - 搜索引擎（如谷歌、百度）\n   - 学术数据库（如PubMed、IEEE Xplore）\n   - 企业文档管理系统\n   - 智能助手（如Siri、Alexa）\n\n4. **发展趋势**：\n   - **大数据**：处理海量信息，提升检索效率。\n   - **人工智能**：机器学习改进准确性。\n   - **自然语言处理**：理解查询意图，超越关键词匹配。\n   - **个性化推荐**：基于用户行为提供相关结果。\n\n5. **与数据挖掘的区别**：\n   - 数据挖掘侧重发现模式和知识，而信息检索侧重找到特定信息。\n\n6. **实际应用与挑战**：\n   - 例子：谷歌搜索通过索引、PageRank和相关性算法提供结果。\n   - 挑战包括处理大量信息、提高准确性、应对多样化需求和防止信息过载。\n\n信息检索是一个复杂且多学科交叉的领域，需要结合计算机科学、信息学和人工智能等知识，不断学习和适应新技术以应对挑战。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None), stop_reason=None)
], created=1739935763, model='DeepSeek-R1-Distill-Qwen-32B', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=1089, prompt_tokens=9, total_tokens=1098, completion_tokens_details=None, prompt_tokens_details=None))
```

---

### 2.4 高阶请求：超参数设置

在使用 OpenAI 的 API 或类似的接口时，通常可以通过传递额外的参数来控制模型的行为。这些参数被称为“超参数”，它们可以影响生成文本的质量、多样性和长度等特性。以下是一些常见的超参数及其作用：


#### 常见超参数
1. **`temperature`**  
   - **作用**: 控制生成文本的随机性或创造性。
   - **取值范围**: 0 到 2（通常推荐 0 到 1）。
     - `0`: 完全确定性输出，模型会选择概率最高的词。
     - `1`: 默认值，平衡创造性和确定性。
     - `>1`: 更具创造性，但也可能更不稳定。
   - **示例**:  
     ```python
     temperature=0.7
     ```

2. **`top_p`** (也叫核采样, nucleus sampling)  
   - **作用**: 控制生成文本时选择词汇的概率累积阈值。
   - **取值范围**: 0 到 1。
     - `1`: 使用所有可能的词汇。
     - `<1`: 只考虑累积概率达到该阈值的词汇。
   - **示例**:  
     ```python
     top_p=0.9
     ```

3. **`max_tokens`**  
   - **作用**: 控制生成文本的最大长度（以 token 数为单位）。
   - **取值范围**: 正整数。
     - 较小值: 生成较短的文本。
     - 较大值: 允许生成较长的文本。
   - **示例**:  
     ```python
     max_tokens=100
     ```

4. **`presence_penalty`**  
   - **作用**: 鼓励模型生成新话题或避免重复已有内容。
   - **取值范围**: -2 到 2。
     - 负值: 鼓励重复已有内容。
     - 正值: 鼓励生成新内容。
   - **示例**:  
     ```python
     presence_penalty=0.5
     ```

5. **`frequency_penalty`**  
   - **作用**: 控制生成文本中重复词汇的频率。
   - **取值范围**: -2 到 2。
     - 负值: 允许更多重复。
     - 正值: 减少重复。
   - **示例**:  
     ```python
     frequency_penalty=0.8
     ```

6. **`stop`**  
   - **作用**: 指定生成文本的停止条件（遇到某些字符串时停止生成）。
   - **取值范围**: 字符串或字符串列表。
   - **示例**:  
     ```python
     stop=["\n", "END"]
     ```

7. **`n`**  
   - **作用**: 指定生成的候选回复数量。
   - **取值范围**: 正整数。
   - **示例**:  
     ```python
     n=3
     ```

8. **`stream`**  
   - **作用**: 是否以流式方式返回结果。
   - **取值范围**: `True` 或 `False`。
     - `True`: 流式返回生成结果。
     - `False`: 一次性返回完整结果。
   - **示例**:  
     ```python
     stream=True
     ```

9. **`logit_bias`**  
   - **作用**: 对特定词汇的生成概率进行调整。
   - **取值范围**: 字典形式，键为 token ID，值为偏移量（正负均可）。
   - **示例**:  
     ```python
     logit_bias={1234: 5, 5678: -10}
     ```

10. **`best_of`** (已废弃)  
    - **作用**: 在内部生成多个候选后选择最佳结果（已被 `n` 参数取代）。



以下是包含多个超参数的完整请求示例：
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
    base_url="http://127.0.0.1:8000/v1",
)

chat_outputs = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {"role": "user", "content": "请写一首关于秋天的诗。"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,
    presence_penalty=0.5,
    frequency_penalty=0.8,
    stream=False,
)

print(chat_outputs)
```


#### 注意事项
1. **参数冲突**: `temperature` 和 `top_p` 不建议同时设置，因为它们的作用有重叠。
2. **性能影响**: 设置较大的 `max_tokens` 或较高的 `n` 会增加计算资源消耗和响应时间。
3. **模型支持**: 不同模型可能对超参数的支持程度不同，请参考具体模型的文档。




## 3. 注意事项

1. **API Key**: 当前服务对 API Key 的验证仅用于接口参数校验，可以随意填写（如 `sk-xxx`）。
2. **最大上下文长度**: 根据模型配置，`max_model_len` 参数限制了输入和输出的总 token 数，请合理设置 `max_tokens` 参数。
3. **性能优化**: 如果需要更高的吞吐量，可以通过调整请求参数（如 `temperature` 和 `max_tokens`）来优化性能。



