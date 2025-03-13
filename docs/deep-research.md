## DeepResearch功能

DeepResearch 框架通过分层查询、递归迭代以及智能决策等步骤，实现深度信息搜索和处理。本流程主要包含以下关键步骤：

1. 意图理解（Intent Understanding）
用户输入查询后，系统会将其解析为多个子查询，以便更精确地理解用户需求。

2. 处理条件判断
系统会根据如下条件判断是否继续执行：
   1. **Token 预算是否超出**
   2. **动作深度是否超出**
  >如果满足上述条件，则终止查询并直接返回答案；否则进入递归执行步骤。

3. 递归执行步骤
在递归执行过程中，系统执行信息检索、模型推理及上下文处理等任务
**信息检索**
- **获取当前问题**
- **构建问题执行序列**
- **递归遍历**
- **深度优先搜索**
-**模型推理**
  >系统进行模型推理，通过系统提示和上下文理解来判断下一步动作。
4. 动作类型判定
根据推理结果，系统决定下一步执行的动作类型：
- **answer**：回答动作
- **reflect**：反思动作
- **search**：搜索动作
- **read**：阅读动作
- **coding**：代码动作

  >这些动作会影响上下文，并不断更新系统状态。

5. 结果反馈
根据最终的动作类型，系统执行相应的任务，并将结果返回给用户，完成整个流程。

DeepResearch流程示意图如下：

![DeepSearch.png](../resources/DeepSearch.png)

运行cli工具：
```bash
cd trustrag/modules/deepsearch
cp .env.example .env #配置LLM API以及搜索
python pipeline.py
```

功能视频演示：

https://www.bilibili.com/video/BV1iLQKYmE1E/?vd_source=7cccd256b3a180013d7510536781e319



## TODO

- [] 大模型反思实现
- [] Gradio集成
- [] Fastapi实现
