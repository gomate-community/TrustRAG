# 导入必要的库
import re  # 用于正则表达式操作

# # 定义目标URL
# url = 'https://finance.sina.com.cn/stock/stockzmt/2025-03-23/doc-ineqqwqv5072383.shtml#/'
#
# # 发送HTTP GET请求以获取网页内容
# response = requests.get(url)
#
# # 检查HTTP请求是否成功
# if response.status_code == 200:
#     # 使用BeautifulSoup解析HTML内容
#     soup = BeautifulSoup(response.content, 'html.parser')
#
#     # 提取网页中的所有文本内容
#     text = soup.get_text()
# else:
#     # 如果请求失败，打印错误信息
#     print(f"Error: {response.status_code}")
# print(text)

with open('../../data/docs/新浪新闻.txt', 'r', encoding='utf-8') as file:
    text = file.read()
# 使用正则表达式将文本按句子分割
# 分割规则：在中文句号、问号、感叹号或引号后跟随的空白字符作为分隔符
single_sentences_list = re.split(r'[。；？！\n]+', text)

# 过滤空句子
single_sentences_list = [sent for sent in single_sentences_list if sent.strip()]

# 打印分割后的句子总数
print(f"{len(single_sentences_list)} sentences were found")
print(single_sentences_list)
# 将每个句子与其索引组成字典列表
sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]

# 打印前6个句子及其索引

print(sentences)


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
print(sentences)

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
# 以下将输出类似如下内容
# [np.float64(0.004011887944849635),
# np.float64(0.5338725292990281),
# np.float64(0.19967922560090334)]

# 可视化向量的距离
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.plot(distances)
plt.show()



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

# plt.title("基于文章内嵌入断点的块")
plt.xlabel("文章内的句子索引（句子位置）")
plt.ylabel("连续句子的余弦距离")
plt.show()

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
