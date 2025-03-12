

# markdown解析
源码：[markdown_parser.py](../trustrag/modules/document/markdown_parser.py)

最近遇到几个伙伴关于markdown解析的问题，都是比较偏向于实际使用场景的，这里我们一开始我们去做markdown文件解析会自觉的会困在一个陷阱，就是：

>我们想把Markdown文件解析效果想的太过理想，会不自觉的与实际生产稳当绑定一起，可能想把Markdown解析数据转成树结构更合理些，但同时考虑内容各式各样，那么这个时候很难下手，不知道怎么去写，常常思考过了半个小时后一行代码也没有写出来。

下面不妨我们尝试把Markdown解析做的**更通用一些**，其他文件类型解析也是这样的套路
>基本上是“File”->"Document"->"Paragraph"-"Chunk"

针对不同类型的知识，我们解析做的效果尽量是将检索信息喂给大模型的时候，我们解析加工的内容不是那么`狼吞虎咽`，也不是那么`细嚼慢咽`

Markdown是带有标题标签的，比如一级标题`#`,二级标题`##`等等，我们可以根据这些标签进行识别段落以及切片。

我们下面采用一个思路，大致是首先识别输出标题以及标题下面对应的内容，然后在标题对应内容内部切片，切片的时候同时保证语义完整。具体做法：

- 能够正确加载解析md文件，识别对应节点类型
- 识别合并出一级标题以及一级标题对应的内容，得到的结果我们称之为段落`paragraph`
- 然后我们在段落内部按照切块算法进行切片，得到的结果我们称之为`chunk`,注意我们采用固定窗口大小的方法，同时需要保证语义的完整性。


## 第一步：Mardkdown文件解析
下面是用langchain解析Markdown例子来做抛砖引玉，例子我们直接参考官方文档[https://python.langchain.com/v0.2/docs/how_to/document_loader_markdown/](https://python.langchain.com/v0.2/docs/how_to/document_loader_markdown/)加载一个本地md文件


```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "../data/docs/md文件3-基础知识.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
# assert len(data) == 1
# assert isinstance(data[0], Document)
# readme_content = data[0].page_content
# print(readme_content[:250])
```


```python
data,len(data)
```
  ![](https://i-blog.csdnimg.cn/direct/4490ae07d8ca4e2faabb88b3032a3154.png)




上面代码是直接加载整个markdown文件，然后把整个文件内容封装成Document对象

如果解析每个标题节点，我们可以使用，可以使用下面一行代码轻松识别所有节点元素，只需要加个参数`mode="elements"`。


```python
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")

data = loader.load()
print(f"Number of documents: {len(data)}\n")

for document in data[:3]:
    print(f"{document}\n")
```

    Number of documents: 52
    
    page_content='这是一篇关于5G基础知识的文章' metadata={'source': '../data/docs/md文件3-基础知识.md', 'languages': ['zho'], 'file_directory': '../data/docs', 'filename': 'md文件3-基础知识.md', 'filetype': 'text/markdown', 'last_modified': '2025-01-08T15:30:41', 'category': 'UncategorizedText', 'element_id': '6b9573733eccc2d1ec188f874e92b0ac'}
    
    page_content='5G基础知识' metadata={'source': '../data/docs/md文件3-基础知识.md', 'category_depth': 0, 'languages': ['zho'], 'file_directory': '../data/docs', 'filename': 'md文件3-基础知识.md', 'filetype': 'text/markdown', 'last_modified': '2025-01-08T15:30:41', 'category': 'Title', 'element_id': '89a807e5cbb5d051e8eacfc9f38bcf41'}
    
    page_content='5G背景' metadata={'source': '../data/docs/md文件3-基础知识.md', 'category_depth': 1, 'languages': ['zho'], 'file_directory': '../data/docs', 'filename': 'md文件3-基础知识.md', 'filetype': 'text/markdown', 'last_modified': '2025-01-08T15:30:41', 'parent_id': '89a807e5cbb5d051e8eacfc9f38bcf41', 'category': 'Title', 'element_id': 'ce0a35331cd9eeb139fb694479abad9e'}
    


- 这是一个元素的示例输出：
     - `page_content`：元素的内容，这里是 `5G基础知识`。
     - `metadata`：元数据，包含以下信息：
       - `source`：文件路径。
       - `category_depth`：分类深度（这里是 0，代表是1级标题）。
       - `languages`：语言（这里是中文 `zho`）。
       - `file_directory`：文件所在目录。
       - `filename`：文件名。
       - `filetype`：文件类型（这里是 Markdown）。
       - `last_modified`：最后修改时间。
       - `category`：元素类别（这里是 `Title`，表示标题）。
       - `element_id`：元素的唯一 ID。

本来还想从头识别标题内容，现在我们直接借花献佛，使用上面结果中的参数`category_depth`来完成第二步的标题段落的识别和合并，我们这里是采用一级标题来做


```python
data[:3]
```



```
    [Document(metadata={'source': '../data/docs/md文件3-基础知识.md', 'languages': ['zho'], 'file_directory': '../data/docs', 'filename': 'md文件3-基础知识.md', 'filetype': 'text/markdown', 'last_modified': '2025-01-08T15:30:41', 'category': 'UncategorizedText', 'element_id': '6b9573733eccc2d1ec188f874e92b0ac'}, page_content='这是一篇关于5G基础知识的文章'),
     Document(metadata={'source': '../data/docs/md文件3-基础知识.md', 'category_depth': 0, 'languages': ['zho'], 'file_directory': '../data/docs', 'filename': 'md文件3-基础知识.md', 'filetype': 'text/markdown', 'last_modified': '2025-01-08T15:30:41', 'category': 'Title', 'element_id': '89a807e5cbb5d051e8eacfc9f38bcf41'}, page_content='5G基础知识'),
     Document(metadata={'source': '../data/docs/md文件3-基础知识.md', 'category_depth': 1, 'languages': ['zho'], 'file_directory': '../data/docs', 'filename': 'md文件3-基础知识.md', 'filetype': 'text/markdown', 'last_modified': '2025-01-08T15:30:41', 'parent_id': '89a807e5cbb5d051e8eacfc9f38bcf41', 'category': 'Title', 'element_id': 'ce0a35331cd9eeb139fb694479abad9e'}, page_content='5G背景')]
```


下面我们也给出基于正则来识别markdown中的文件，具体做法如下：


```python
import re
from langchain_core.documents import Document

def parse_markdown_to_documents(content):
    # 正则表达式匹配Markdown标题
    heading_pattern = re.compile(r'^(#+)\s*(.*)$', re.MULTILINE)
    
    # 存储解析结果
    documents = []
    
    # 初始深度
    current_depth = 0
    
    # 分割内容
    sections = content.split('\n')
    
    for section in sections:
        # 检查是否是标题
        heading_match = heading_pattern.match(section)
        if heading_match:
            # 计算标题的深度
            current_depth = len(heading_match.group(1)) - 1
            # 提取标题内容
            page_content = heading_match.group(2).strip()
            # 添加到结果中
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={"category_depth":current_depth}
                )
            )
        else:
            # 如果不是标题，且内容不为空，则添加到结果中
            if section.strip():
                documents.append(
                    Document(page_content=section.strip(),metadata={})
                )
    return documents

# 示例调用
with open(markdown_path,"r",encoding="utf-8") as f:
    content=f.read()
```


```python
parsed_documents = parse_markdown_to_documents(content)
merge_title_content(parsed_documents)
```



```
    [{'title': '', 'content': '这是一篇关于5G基础知识的文章'},
     {'title': '# 5G基础知识',
      'content': '## 5G背景\n流量指数级增长、人与人的通信过度到人与物和物与物、应用场景多样化这三方面催生了 5G。\n2G、3G、4G 主要解决人与人之间的通信，5G 不仅要解决人与人之间的通信，而且要解决人与物、物与物之间的通信，从而达成万物互联的目的。\n## 5G新技术三个特征\n### 新核心网\n4G 核心网就像是在一块空地上建好的房子，每个房间都有其固定用途，不能用作他用。而 5G 核心网只提供了地皮和一些标准件，我们可以像搭积木一样随心所欲的按照自己喜欢的房间样式自由组合；\n### 新传输网\nSPN 具备前传、中传和回传的端到端组网能力，支持端到端网络硬切片能力，满足动态灵活连接需求；\n### 新无线网\n频谱效率提升 3 倍，连接数密度提升 10 倍，峰值速率提升 10 倍，空口时延迟为降低到原来的 1/10；\n## 5G三大场景\neMMB 增强移动宽带\nuRLLC 超高可靠性与超低时延业务\nmMTC 海量物联网通信\n用户体验速率是 4G 的 10-100 倍，每平方公里的链接数是 4G 的 10 倍，典型场景的时延可低至 10ms 以内。\n## 5G与4G的对比\n5G 网络像一个魔方，它可以根据需求不停变形，从而满足个人或者企业不同的个性化需求。\n4G 网络千人一面，5G 网络千人千面。\n## 5G与WiFi的对比\nWiFi 秉承互联网“始终尽力而为”的传统：“不管什么情况，我尽可能给您快，不保障一直够快，偶尔卡死，您见谅！”。\n5G 则秉承着更有保障的 QoS 承诺，紧急业务时延一定可控，非紧急业务尽力而为。不同WiFi 需要手动连接，5G 无缝切换。\n## 5G与有线的对比\n剪掉辫子，随时随地不受限。有线易磨损，改造成本高。5G 无线让最后一公里的接入更灵活。'},
     {'title': '# 5G 专网知识',
      'content': '## 基本概念\n基于授权频谱，为专有行业客户提供服务范围、网络能力、隔离度可定制的 5G 通信服务。\n## 专网模式\n优享：复用大网资源，通过配置 5QI、DNN、网络切片等保障行业用户的 QOS。\n专享：无线侧按需补点增强覆盖，PRB 资源预留，核心网用户面UPF 和边缘计算MEP 设备按需下沉。\n尊享：基站和频率资源独享实现高隔离高可靠，核心网用户面专用，控制面资源按需提供。\n## BAF商业模式\nBAF 网络服务模式基于 5G 产品清单，包括 3 项基础架构（B），12 项增值功能（A），个性化组合(Flexible)，满足客户的个性化需求实现让客户“按单点菜”。\n## 用户面功能 UPF\n可以把UPF 看成一个路由器主要实现分流，5GC 核心网上面的网元 SMF 制定分流策略，\n通过 DNN 或ULCL 的方式，需要到达内网的数据可以直接进行 MEC，MEC 处理后，进入用户的内网，这样的数据传输环节将大大减小，适合许多时延要求较高的应用，同时基于 5G 的特性，可以传输类似于视频之类的大数据业务。\n## 边缘计算 MEC\n边缘计算最常用的比喻就是章鱼的神经系统。它的大脑作为中央节点只处理 40%的信息，主要负责总体协同，剩余的 60%的信息则由 8 条触手（相当于边缘节点）就近处理。\n## 网络切片\n说 4G 网络是一把刀，足可削铁如泥、吹毛断发。那么，5G 网络就是一把瑞士军刀，灵活方便、多功能用途。每个虚拟网络就像是瑞士军刀上的钳子、锯子一样，具备不同的功能特点，面向不同的需求和服务。\n## 边缘云\n把云计算看作是大脑，那么边缘计算就像是大脑输出的神经触角，这些触角连接到各个终端运行各种动作。'},
     {'title': '# 5G 双域专网',
      'content': '## 基本概念\n以 5G 专网为基础提供服务于 5G 用户的 2B2C 双域网络模式，可满足企业用户“不换卡、不换号、无感知切换”，随时随地、安全快捷访问办公内网和互联网，助力企业办公移动化、灵活化。\n## 应用场景\n（1） 强调“广域接入”，移动终端“不换卡不换号”、强调支持特定号码全国漫游自由接入双域专网；\n（2） 强调“局域接入”，移动终端“不换卡不换号”、强调支持特定号码、特定区域自由接入双域专网，要求出区域禁止访问专网。\n（3） 强调“局域接入”，移动终端“不换卡不换号”、强调支持非特定号码、特定区域自由接入双域专网，要求出区域禁止访问专网。\n## 实现方案\n（1）通用 DNN+ULCL；（2）专用 DNN+IP 分流；（3）通用 DNN+ULCL+专用 DNN（可选）；（4）通用 DNN+专用 DNN+IP 分流。'}]
```


## 第二步：标题段落内容识别合并


```python
def merge_title_content(data):
    merged_data = []
    current_title = None
    current_content = []

    for document in data:
        metadata = document.metadata
        category_depth = metadata.get('category_depth', None)
        page_content = document.page_content

        # 如果 category_depth 为 0，表示遇到一级标题
        if category_depth == 0:
            # 如果当前标题不为空，表示已经收集了一个完整的一级标题及其内容
            if current_title is not None:
                # 将当前标题和内容合并为一个字符串，并添加到 merged_data 中
                merged_content = "\n".join(current_content)
                merged_data.append({
                    'title': current_title,
                    'content': merged_content
                })
                # 重置当前标题和内容
                current_content = []

            # 更新当前标题，并根据 category_depth 添加 Markdown 标记
            current_title = f"{'#' * (category_depth + 1)} {page_content}"

        # 如果 category_depth 不是 0，表示是正文或其他内容
        else:
            # 如果当前标题为空，表示一开始就是正文
            if current_title is None:
                merged_data.append({
                    'title': '',
                    'content': page_content
                })
            # 一级标题之外的标题，比如二级、三级等
            elif category_depth is not None:
                # 添加 Markdown 标题标记
                current_content.append(f"{'#' * (category_depth + 1)} {document.page_content}")
            else:
                # 将内容添加到当前内容列表中
                current_content.append(page_content)

    # 处理最后一个标题及其内容
    if current_title is not None:
        merged_content = "\n".join(current_content)
        merged_data.append({
            'title': current_title,
            'content': merged_content
        })

    return merged_data
```


```python
# 假设 data 是已经加载的文档列表
merged_data = merge_title_content(data)

# 输出合并后的标题和内容
for item in merged_data:
    print(f"一级标题: {item['title']}")
    print(f"段落内容: {item['content']}\n")
    print("==="*10)
```

![](https://i-blog.csdnimg.cn/direct/1533d8accc78435aae4d72a2c32ce1f2.png)



```python
len(merged_data)
```

 >   4
## 第三步：内容切片

对于内容切片，常用的文本分块方法包括：固定大小分块、基于 NTLK 分块、特殊格式分块、深度学习模型分块、智能体式分块。

下面我们采用常用的固定大小分块在`段落内容内`进行切块进行切块,为了保证语义的完整性，首先对段落内容进行句子切片，然后按照chunk_size窗口大小对句子进行合并，如果不满足窗口大小就添加下一个句子，一直到大于等于窗口大小就停止。


```python
import sys
sys.path.append('/Users/yanqiang/Projects/TrustRAG')
!pwd
```

    /Users/yanqiang/Projects/TrustRAG/notebooks



```python
from trustrag.modules.document.chunk import TextChunker
```


```python
tc=TextChunker()
```


```python
paragraphs=[item["title"]+"\n"+item["content"] for item in merged_data]
chunks=[]
for para in paragraphs:
    chunks.extend(tc.chunk_sentences(para,chunk_size=256))
```


```python
len(chunks)
```




 >   7


```python
chunks
```



