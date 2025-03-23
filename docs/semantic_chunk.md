
在RAG（Retrieval-Augmented Generation）中，chunk是个关键步骤。它的核心目标，就是把语义相近的内容放在一起，语义不同的内容拆开，这样后续的检索（retrieve）和重排序（rerank）才能更有效。  

举个例子：  

> 今天天气很好，我和小明在一起打篮球。  
> 隔壁老王在家里看电视。  
> 小明的妈妈在家里做晚饭，晚上我去小明家吃饭。  

这段话其实表达了三个完全不同的意思，最理想的chunk方式，就是精准地把这三个部分分开，互不干扰。  

## 语义切块思路

### 1. 传统Chunk方式的局限  

目前常见的chunk方法主要有以下几种：  

- **按固定长度切分**：比如每256个token切一块，但这样可能会把同一语义的内容拦腰截断。  
- **按标点符号切分**：比如遇到句号、分号、换行符就分割，看起来合理，但如果一个观点跨了两句话，这种方法就显得不够智能。  

有没有更符合语义的切割方式？当然有！**Semantic Chunk（语义切块）** 就是专门为了解决这个问题的！LangChain 和 LlamaIndex 已经实现了这个功能，但它们默认用的是 OpenAI 的 embedding API，不仅收费，还可能有访问限制。所以，自己实现 Semantic Chunk 还是很有必要的！  

### 2. 语义切块的基本思路  

**目标：把相似语义的句子聚合在一起，不相关的句子拆开。**  

第一步，我们还是先**按标点符号切分**，形成多个句子：  
```
sen1, sen2, sen3, ..., senN
```
接下来，我们计算每个句子的 embedding，然后就可以用聚类方法来分组了。  

### 3. 改进方案：滑动窗口+语义相似度  

问题来了——文章里语义相似的句子通常都挨得比较近，所以**简单的聚类方法可能会把远距离的句子放在一起**，这样就不合理了。怎么优化呢？  

我们可以用**滑动窗口**的方法，按局部语义变化来切割：  

1. **从头开始，逐步合并句子**，比如：  
   ```
   sen1 → sen1 + sen2 → sen1 + sen2 + sen3
   ```
   以此类推，每次新增一个句子，形成 `combined_sentence1`、`combined_sentence2`...  

2. **计算相邻combined_sentence的语义相似度**，如果相似度突然下降，说明新加入的句子跟前面的内容语义差别大，应该从这里切块。  

举个例子：  
```
sen1+sen2 和 sen1+sen2+sen3 的相似度很高  
sen1+sen2+sen3 和 sen1+sen2+sen3+sen4 的相似度突然降低  
```
那么我们就可以认为，`sen1 ~ sen3` 是一个chunk，`sen4` 需要单独开一个chunk。  


我们用一张图来表示大致原理：

![](https://i-blog.csdnimg.cn/direct/9259ef51121245e7a77693e7eb569aef.png)
## 语义切块实现

我们找一段文本来当做测试样例
![](https://i-blog.csdnimg.cn/direct/7ba74f8c929f42848cd4e2da854d1f88.png)


```
# 导入必要的库
import requests  # 用于发送HTTP请求
from bs4 import BeautifulSoup  # 用于解析HTML内容
import re  # 用于正则表达式操作

# 定义目标URL
url = 'https://finance.sina.com.cn/stock/stockzmt/2025-03-23/doc-ineqqwqv5072383.shtml#/'

# 发送HTTP GET请求以获取网页内容
response = requests.get(url)

# 检查HTTP请求是否成功
if response.status_code == 200:
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.content, 'html.parser')

    # 提取网页中的所有文本内容
    text = soup.get_text()
else:
    # 如果请求失败，打印错误信息
    print(f"Error: {response.status_code}")

# 使用正则表达式将文本按句子分割
# 分割规则：在中文句号、问号、感叹号或引号后跟随的空白字符作为分隔符
single_sentences_list = re.split(r'(?<=[。？！」])\s+', text)

# 打印分割后的句子总数
print(f"{len(single_sentences_list)} sentences were found")

# 将每个句子与其索引组成字典列表
sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]

# 打印前6个句子及其索引

print(sentences[:6])
```
26 sentences were found
![](https://i-blog.csdnimg.cn/direct/e089053185aa4cef8da00bde5dd9997f.png)
**有个问题，单独用一个句子来做相似度关联可能有点弱，所以我们需要建立一个缓冲区，也就是一个句子加上下一个句子来判断相似度，这样尽量减小单个句子语义的偏差。**

```
# 定义一个函数，用于将句子列表中的句子按照指定的缓冲区大小进行组合
def combine_sentences(sentences, buffer_size=1):
    """
    参数:
    sentences (list): 包含句子及其索引的字典列表，格式为 [{'sentence': '句子内容', 'index': 索引}]。
    buffer_size (int): 用于定义前后文句子数量的缓冲区大小，默认值为1。

    返回值:
    list: 更新后的句子列表，每个句子字典新增一个键 'combined_sentence'，表示组合后的句子。
    """

    # 遍历句子列表中的每个句子
    for i in range(len(sentences)):

        # 初始化一个空字符串，用于存储组合后的句子
        combined_sentence = ''

        # 将当前句子之前的句子（基于缓冲区大小）添加到组合句子中
        for j in range(i - buffer_size, i):
            # 确保索引 j 不为负数，避免访问列表时超出范围
            if j >= 0:
                # 将索引 j 对应的句子追加到组合句子中，并用空格分隔
                combined_sentence += sentences[j]['sentence'] + ' '

        # 将当前句子添加到组合句子中
        combined_sentence += sentences[i]['sentence']

        # 将当前句子之后的句子（基于缓冲区大小）添加到组合句子中
        for j in range(i + 1, i + 1 + buffer_size):
            # 确保索引 j 不超出句子列表的范围
            if j < len(sentences):
                # 将索引 j 对应的句子追加到组合句子中，并用空格分隔
                combined_sentence += ' ' + sentences[j]['sentence']

        # 将组合后的句子存储到当前句子字典中，键名为 'combined_sentence'
        sentences[i]['combined_sentence'] = combined_sentence

    # 返回更新后的句子列表
    return sentences

# 调用 combine_sentences 函数，对句子列表进行处理
sentences = combine_sentences(sentences)

# 打印处理后的前3个句子及其组合结果
print(sentences[:3])
```

![](https://i-blog.csdnimg.cn/direct/807858d5de0c41e09fc3655255209b2d.png)

使用嵌入模型对每个句子进行向量化并计算余弦相似度的代码

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_name_or_path='G:/pretrained_models/mteb/bge-m3')
embeddings = model.encode([x['combined_sentence'] for x in sentences])
print(embeddings)
for i, sentence in enumerate(sentences):
    sentence['combined_sentence_embedding'] = embeddings[i]


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']

        # 计算余弦相似度
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # 将余弦相似度转换为余弦距离（余弦距离 = 1 - 余弦相似度）
        distance = 1 - similarity

        # 将余弦距离添加到列表中
        distances.append(distance)

        # 将距离保存到字典中
        sentences[i]['distance_to_next'] = distance

    # 如果需要处理最后一个句子的情况（可选）
    # sentences[-1]['distance_to_next'] = None  # 或者设置默认值

    return distances, sentences


distances, sentences = calculate_cosine_distances(sentences)
print(distances[:3])
```
输出如下：
```python
[[-0.06047463 -0.02308427 -0.05546052 ...  0.01303777  0.04497439
  -0.01352415]
 [-0.05727438 -0.02179192 -0.05408231 ...  0.01058046  0.04768301
  -0.01813268]
 [-0.05887977  0.00261552 -0.06078574 ... -0.00508141  0.04915523
  -0.01307676]
 ...
 [-0.07664938  0.0410533  -0.02591059 ... -0.01490396  0.05444359
  -0.00391266]
 [-0.07636633  0.05034129 -0.01471427 ... -0.02054079  0.04689409
  -0.0023479 ]
 [-0.06998076  0.03120246 -0.01492447 ...  0.00981126  0.02617716
   0.00134683]]
[0.008807897567749023, 0.32371556758880615, 0.166284441947937]
```
可视化向量的距离
```python
# 可视化向量的距离
import matplotlib.pyplot as plt

plt.plot(distances)
plt.show()
```

向量距离的可视化如下：

![](https://i-blog.csdnimg.cn/direct/817229789d0247048d742515ef5789c7.png)
按余弦距离分割文本并进行可视化

```python
# 根据文章的余弦距离，将文章内的句子分成“块”，并可视化这些区间。
import numpy as np

plt.plot(distances)

y_upper_bound = .2
plt.ylim(0, y_upper_bound)
plt.xlim(0, len(distances))

# 需要确定被视为异常值的距离阈值
# 这里使用numpy的percentile()函数
breakpoint_percentile_threshold = 90
breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)  # 如果希望得到更多的块，请降低百分位数截止值
plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')

# 接下来，检查超过此阈值的距离有多少个
num_distances_above_threshold = len([x for x in distances if x > breakpoint_distance_threshold])  # 超过阈值的距离数量
plt.text(x=(len(distances) * .01), y=y_upper_bound / 50, s=f"{num_distances_above_threshold + 1} 块")

# 接下来，获取超过阈值的距离的索引。这将帮助确定文本应该分割的位置
indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]  # 列表中超过阈值的点的索引

# 开始着色和文本标注
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, breakpoint_index in enumerate(indices_above_thresh):
    start_index = 0 if i == 0 else indices_above_thresh[i - 1]
    end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

    plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
    plt.text(x=np.average([start_index, end_index]),
             y=breakpoint_distance_threshold + (y_upper_bound) / 20,
             s=f"块 #{i}", horizontalalignment='center',
             rotation='vertical')

# 最后一个断点到数据集末尾添加着色
if indices_above_thresh:
    last_breakpoint = indices_above_thresh[-1]
    if last_breakpoint < len(distances):
        plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
        plt.text(x=np.average([last_breakpoint, len(distances)]),
                 y=breakpoint_distance_threshold + (y_upper_bound) / 20,
                 s=f"块 #{i+1}",
                 rotation='vertical')

plt.title("基于文章内嵌入断点的块")
plt.xlabel("文章内的句子索引（句子位置）")
plt.ylabel("连续句子的余弦距离")
plt.show()
```

![](https://i-blog.csdnimg.cn/direct/64c05ff88e3d4ae0a7f48214adff3781.png)
我们最后来实现分块
```
# 初始化开始索引
start_index = 0

# 存储分组后的句子的列表
chunks = []

# 根据断点对句子进行分片
for index in indices_above_thresh:
    # 结束索引是当前断点
    end_index = index

    # 从当前开始索引到结束索引之间的句子进行切片
    group = sentences[start_index:end_index + 1]
    # 将每个句子合并成一个字符串
    combined_text = ' '.join([d['sentence'] for d in group])
    # 将合并后的文本添加到块列表中
    chunks.append(combined_text)

    # 更新开始索引以处理下一个组
    start_index = index + 1

# 最后一个组（如果有剩余的句子）
if start_index < len(sentences):
    combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
    chunks.append(combined_text)

# grouped_sentences（或chunks）中存储了分割后的句子

for i, chunk in enumerate(chunks):
    buffer = 200

    print(f"块 #{i}")
    print(chunk[:buffer].strip())
    print("\n")

```

分块结果如下：
```text
块 #0
被这一轮恒生科技大行情甩下车，百度，会不会下了牌桌 炒股就看金麒麟分析师研报，权威，专业，及时，全面，助您挖掘潜力主题机会


块 #1
来源：基本面力场 3月18日，百度集团（9888.HK）以一根涨幅超过12%的大阳线，创出股价近期的小高点，大有向上突破的趋势和节奏 也是同一天，“开盒”事件全面引发舆情关注，百度发表声明紧急撇清关系，强调涉事数据并非来自公司内部，而是源自海外的“社工库” 但是这似乎并未能挽救市场对于百度股价的信心，次日也即3月19日，百度股价大幅下跌3.97%，随后连续两日持续大跌，跌幅分别高达5.44%和2.


块 #2
1、起个大早、赶个晚集：百度的创新力让人失望 最近，零一万物创始人兼CEO李开复公开表示，DeepSeek的崛起正彻底颠覆人工智能行业，并可能对OpenAI形成重大冲击 他还预测中国市场最终可能仅剩DeepSeek、阿里巴巴和字节跳动三家主要AI模型公司，其中DeepSeek势头强劲 引起力场君关注的是，李开复预判的国内大模型三巨头，没有百度，和他的文心一言 在ChatGPT横空出世之后，百度是最


块 #3
反正力场君身边的小伙伴，基本投入到了deepseek和腾讯混元的怀抱


块 #4
对于百度而言，这不是起个大早、赶个晚集吗 关键是，这不是第一次了，力场君仍记得，百度也是国内最早布局智能驾驶的大厂，由百度的“萝卜快跑”运营的无人出租车，还曾引发武汉市出租车、网约车群体的声讨 但步入到2024年下半年和今年初，智驾的牌桌上，以华为和比亚迪(372.000, -27.99, -7.00%)为双雄、一众新能源车自研智驾系统众星捧月，“萝卜快跑”似乎已淡出大众的视野 力场君特意翻阅了一


块 #5
同时，科技活动更需遵循伦理道德准则，防止滥用技术，保护个人隐私和数据安全 不论是早年的魏则西事件，还是刚刚发生的开盒事件，无疑都是对科技向善理念的挑战 得道多助、失道寡助，丧失与人为善的科技，又怎么会有持续的生命力，和增长的想象力 回到业务层面，百度所依仗的搜索业务衍生出的广告服务，仍是核心利润来源，但是在问答大模型快速普及的趋势面前，搜索引擎的重要性光环，正在快速褪去 正如东吴证券(7.980,



```
## 还能怎么优化？  

除了滑动窗口的方法，我们还可以考虑：  

- **动态窗口大小**：短句可能需要更大窗口，长句可能需要更小窗口。  
- **结合TF-IDF或主题建模**：判断内容是否属于同一话题，再做chunk。  
- **引入监督学习**：用已有数据微调一个模型，让它自动学习最佳切割点。  

总之，手撸Semantic Chunk并不难，核心就是**基于语义相似度，找到合理的切割点**。相比传统的固定长度或标点分割，**这种方法更加智能、精准**，对于提升 RAG 的检索效果非常有帮助！  

