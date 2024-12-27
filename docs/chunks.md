## 文本分块

### 切块定义
分块（Chunking）是将整篇文本分成小段的过程。当我们使用LLM embedding内容时，分块可以帮助优化从向量数据库被召回的内容的准确性，因此文本段的质量也是RAG中比较重要的一环。

### 切块作用

切块的主要目的是对上下文中的一个片段进行嵌入的过程中尽可能少的噪声，而还要保证内容仍然与语义相关。



### 常见的chunk切分方法

1、  固定长度切分
● 操作：按照文本的字数或者词数将文本切分为多块。比如可以将文档按照500字切分，切分之后的每个文本块字数为500。

● 优点：简单易实现，可快速处理。

● 缺点：可能会导致上下文断裂，影响重要的语义信息。

2、  基于句子的切分
● 操作：按照句子粒度进行切分，比如以句号、点号等标点符号进行切分

● 优点：该方法能保证每个句子的完整性、上下文连贯性

● 缺点：如果句子过长，可能丢失一些细节。可能切分的不准确，影响检索效果。

3、  滑动窗口切分
● 操作：创建一个重叠的滑动窗口，比如设置窗口大小为500，步长为100。

● 优点：可以减少因固定长度或句子边界切分可能引入的信息丢失问题。

● 缺点：上下文重叠导致信息重复，增加计算量。窗口的开始和结束可能会在句子或短语中间，导致语义不连贯。

4、  基于主题切分
● 操作：通过识别文章主题的变换点进行切分。

● 优点：保持高度的语义连贯性，适用于结构化比较好的文本。

● 缺点：无法处理结构化不足的文本。

5、  基于语义相似度的切分
● 操作：使用模型来评估文本间的语义相似度，并在相似度降低到某个阈值以下时进行切分

● 优点：保持高度语义相似性，优化检索效果

● 缺点：模型准确率要求高

6、按文档结构切分
● 操作：典型的是markdown切分工具，按照文档结构切分

● 优点：语义连贯

● 缺点：有的问题涉及多个部分的内容，可能无法覆盖；生成模型的token数有限制，该切分方式可能不满足token限制；

7、文档块摘要切分
● 操作：切分文档后，使用摘要生成技术来提取每个块的关键信息

● 优点：可以将关键信息精简并保留

● 缺点：摘要生成方法的精度直接影响整体效果

### 分块需要考虑因素：
1、  被索引内容的性质是什么？

是处理较长的文本(书籍或文章)，还是处理较短的内容。不同场景需要的分块策略不同。

2、  不同的embedding模型在不同大小块上的效果不同

3、  查询query的长度和复杂度与块的切分有很大关系

用户输入的查询文件时简短而具体的还是冗长而复杂的。

4、  如何在特定的程序中使用检索结果

比如在LLM中，token长度会限制切块的大小。

**注意：文档切的多，向量就多，导致查询效率变差，语义内聚性也降低。因此，没必要切的时候，尽量别切。但是切的时候也要顶着最大长度切，能有效降低文档切块的数量。**


### 参考资料
[RAG中常见的chunk切分方法](https://www.ctyun.cn/developer/article/551915360890949)