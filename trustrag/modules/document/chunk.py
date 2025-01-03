import re
from tqdm import tqdm
from trustrag.modules.document import rag_tokenizer


class TextChunker:
    def __init__(self, ):
        self.tokenizer = rag_tokenizer

    # def split_sentences(self, text):
    #     # 使用正则表达式按中英文标点符号进行分句
    #     sentence_endings = re.compile(r'([。！？])')
    #     sentences = sentence_endings.split(text)
    #
    #     # 将标点符号和前面的句子合并
    #     sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])] + sentences[2::2]
    #     sentences=[sentence.strip() for sentence in sentences if sentence.strip()]
    #     return sentences

    def split_sentences(self, text):
        # 使用正则表达式按中文标点符号进行分句
        sentence_endings = re.compile(r'([。！？.!?])')
        sentences = sentence_endings.split(text)

        # 将标点符号和前面的句子合并
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if sentences[i]:
                result.append(sentences[i] + sentences[i + 1])

        # 处理最后一个可能没有标点的句子
        if sentences[-1]:
            result.append(sentences[-1])

        # 去除空白并过滤空句子
        result = [sentence.strip() for sentence in result if sentence.strip()]

        return result

    def process_text_chunks(self, chunks):
        processed_chunks = []
        for chunk in chunks:
            # 处理连续的四个及以上换行符
            while '\n\n\n\n' in chunk:
                chunk = chunk.replace('\n\n\n\n', '\n\n')

            # 处理连续的四个及以上空格
            while '    ' in chunk:
                chunk = chunk.replace('    ', '  ')

            processed_chunks.append(chunk)

        return processed_chunks

    def chunk_sentences(self, paragraphs, chunk_size):
        """
        将段落列表按照指定的块大小进行分块。

        首先对拼接的paragraphs进行分句，然后按照句子token个数拼接成chunk

        如果小于chunk_size，添加下一个句子，直到超过chunk_size，那么形成一个chunk
        依次生成新的chuank

        参数:
        paragraphs (list): 要分块的段落列表。

        返回:
        list: 包含分块后的文本的列表。
        """
        # 将段落列表拼接成一个大文本
        text = ''.join(paragraphs)

        # 分句
        sentences = self.split_sentences(text)
        # print(sentences)

        if len(sentences) == 0:
            sentences = paragraphs
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if current_chunk_tokens + len(tokens) <= chunk_size:
                current_chunk.append(sentence)
                current_chunk_tokens += len(tokens)
            else:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_chunk_tokens = len(tokens)

        if current_chunk:
            chunks.append(''.join(current_chunk))
        chunks = self.process_text_chunks(chunks)
        return chunks


if __name__ == '__main__':

    # 示例使用
    paragraphs = ['2020年9期', '总第918期商业研究', '一、研究背景和意义',
                  '2020年新春伊始 ，一场突如其来的疫情打乱了人们欢天喜',
                  '地回家过年的计划 ，节日的气氛被病毒侵肆 ，1月23日，武汉打',
                  '响“封城”第一枪，接着从湖北各市区乃至全国各地纷纷开 启一',
                  '级应急响应 ，全国人民停工 、停产、停学,开始居家隔离 ，春节档电',
                  '影除夕当天宣布延期上映 ，电影院、商场、中小商铺 、外卖、快递',
                  '几乎全部停工 ，居民消费行为呈现出 “宅经济”的特点，经济社会',
                  '运行受到严重冲击 。国家统计局数据显示 ，2020年1月-2月，我',
                  '国社会消费品零售总额同比下降 20.5%。消费不仅是衡量 居民生',
                  '活水平的重要指标 ，同时亦是我国经济增长的主要驱动 力。因',
                  '此，新冠疫情在国内得到基本控制的情况下 ，国家一手紧 抓复工', '复产，一手紧抓内外防控 。',
                  '有学者认为疫情期间居家隔离举措会影响消费者消费理',
                  '念，降低整体消费水平 ，也有学者认为 ，疫情结束以后 ，居民会产',
                  '生“报复性消费 ”，显著提高消费水平 ，更有认为疫情只是暂时 延',
                  '缓了经济社会发展进程 ，长久看来 ，并不会对消费行为造成较 大',
                  '影响。研究新冠肺炎疫情下消费者心理 、消费能力 、消费方式 、消',
                  '费对象等方面的变化有助于判断未来消费行为变动趋势以及 企',
                  '业应对措施 ，有助于充分挖掘国内市场 、重振居民消费 、全面建',
                  '成小康社会以及实现 2020年经济社会发展目标 。',
                  '二、新冠肺炎疫情特点及现状', '新型冠状病毒肺炎 ，简称“新冠肺炎 ”，2020年2月11日，世',
                  '界卫生组织其命名为 “COVID-19 ”。疫情开始于 2019年末，湖北',
                  '省武汉市部分医院陆续发现了多例有华南海鲜市场暴露史的 不',
                  '明原因肺炎病例 ，后经证实为 2019新型冠状病毒感染引起的 急',
                  '性呼吸道传染病 。以发热、乏力、干咳为主要表现 ，感染病人约 有',
                  '1天-14天的潜伏期 ，早期症状与普通感冒相似 ，不易发觉 ，具有',
                  '传播性强、严重程度高的特点 ，死亡率大约为 3%，其传播途径主',
                  '要为直接传播 、气溶胶传播和接触传播等人传人形式 ，呈现聚集',
                  '性感染。目前国内确诊人数超八万人 ，全球病例更达到 288万这',
                  '个骇人听闻的数字 ，全球国家经济受到严重冲击 ，疫情防控更是', '重中之重。',
                  '三、新冠肺炎疫情对居民消费行为的影响', '1.消费者心理',
                  '2020年1月30日，世界卫生组织宣布新冠肺炎疫情为国 际',
                  '关注的突发公共卫生事件 ，当时国外疫情还未完全爆发 ，中国政',
                  '府为了保证人民的安全健康 ，冒着经济社会可能受到重创的 风',
                  '险，出台了一系列 “封城”、“封路”、“停工停产 ”等措施进行 疫情',
                  '防控。居民在居家防控期间 ，安全健康意识显著提高 ，消费者对',
                  '于防控用品的需求增加 ，在进行消费时也更加注重环境的 安全卫生，由于疫情期间出行受限 ，消费由线下进一步向线上转 变，',
                  '居民的精神文化需求进一步增高 ，对于资讯以及社交需求方 面', '消费相应提高 。', '2.消费能力',
                  '由于疫情防控 ，许多人都表示今年过了最长的一个春节假',
                  '期，各企业工厂等不得不停工停产 ，大量从业人员无法复工 ，只',
                  '有基本生活保障收入 ，消费能力显著下降 。疫情期间部分企业经',
                  '营困难，甚至出现资金流断裂 、退出市场的情况 ，大量小型企业',
                  '倒闭或是选择暂时停业 ，更有部分企业有减员计划 ，其余对普通',
                  '员工只发放基本生活保障或按本地最低工资标准发放 ，导致从', '业人员收入和消费能力大大下降 。',
                  '3.消费方式',
                  '受到疫情防控出行限制 ，线下消费受到了抑制 ，在快递、外',
                  '卖等未停工地区 ，居民线上消费意愿强烈 ，主要体现在线上购',
                  '物、线上教育 、网络影视等方面 。居民线上购物大范围替代线下',
                  '购物，例如湖北各市区小区封闭时 ，居民采取线上下单 ，由社区',
                  '工作人员或志愿者等上门送货 。居民居家隔离期间 ，对于科普 、',
                  '文化等服务类消费的意愿普遍增加 ，普遍愿意增加医疗保健 、保', '险等服务消费 。', '4.消费对象',
                  '首先，居民安全健康意识提高 ，对于疫情防控用品 ，例如口',
                  '罩、护目镜、消毒液、酒精等消费大幅度增加 ，对于医疗保健用',
                  '品、保险的消费也有一定程度提高 。2020年1月19日至22日期',
                  '间，京东平台累计销售口罩 1.26亿只、消毒液31万瓶、洗手液',
                  '100万瓶，其中电子体温计 、感冒药、VC泡腾片、护目镜等相关商', '品较平时也有一定程度提高 ，',
                  '其次，居民线上消费对象不断拓展 ，网购对象从标准化程度',
                  '较高和易于快递配送的商品向生鲜 、医药等非标准化和低频商',
                  '品延伸，春节期间 ，京东到家全平台销售额相比去年同期增长',
                  '470%，盒马鲜生日均蔬菜供应量是平时的 6倍，美团外卖慢性处', '方药销量增长 237%。',
                  '第三，居民的旅游支出降低 ，许多消费者由于疫情影响取消',
                  '了春节期间的出行计划 ，旅游行业遭受重大冲击 ，疫情结束后各',
                  '大景点采取一定免费形式吸引消费者出行 ，但是居民旅游总支', '出仍然大幅度降低 。',
                  '最后，居民对于数字文化娱乐服务产品消费显著提高 ，例如',
                  '网络视频会员 、游戏、线上教育 、办公等用户大量增加 ，线上消费',
                  '群体则迅速扩张 ，老年居民和学龄儿童也开始进行线上购物 。', '四、未来居民消费行为变动趋势',
                  '1.线上线下消费融合浅谈新冠肺炎疫情对居民消费行为的影响', '■谭诗怡 湘潭大学商学院',
                  '摘要：随着新冠肺炎疫情突发 ，不仅对中国经济和产业会产生很大影响 ，而且会短期或长期地改变民众的消费行为模式 ，这引',
                  '起了各类政府机构 、工商企业 、平台商家的高度关注 。不仅消费者的消费理念 、消费方式 、消费对象 、消费能力在疫情影响下发生了 一',
                  '系列改变 ，同时餐饮 、住宿、旅游、娱乐等传统服务业也受到了严重冲击 。本文通过网络渠道等途径 ，收集相关资料 ，研究未来居民 消',
                  '费行为的变动趋势分析以及企业对消费者行为模式变化后的应对措施 ，并提出相关国内市场发展建议 。',
                  '关键词：新冠肺炎 ；居民消费 ；消费者行为', '10Copyright©博看网 www.bookan.com.cn. All Rights Reserved.',
                  '商业研究', '目前居民消费由线下向线上转移 ，会形成居民的消费惯性 ，',
                  '在疫情结束以后 ，并不会大幅降低线上消费水平 ，且国内疫情暂',
                  '时处于稳定期 ，后续还存在二次爆发的可能性 ，电影院、KTV、电',
                  '玩城等聚集性娱乐场所也未复工 ，居民日常出行仍然要注意安',
                  '全防护，去公共场所要佩戴口罩 ，在餐厅、医院等人员聚集场所 ，',
                  '排队须有间隔 ，座位要相隔而坐 ，并且还有大量师生还未返校 ，',
                  '学生居家上网课 。因此，复工复产的企业员工消费能力有所回',
                  '暖，但居民生活方面以及精神方面的消费还要依靠线上得以满',
                  '足。线上线下消费融合是未来居民消费变动一大重要趋势 ，不仅',
                  '可以保证一定程度的安全健康 ，还极大地便利了居民消费 、满足', '了他们的需求欲望 。',
                  '2.注重安全健康消费',
                  '居民消费时更加注重安全性以及健康性 ，首先消费者优先',
                  '考虑线上服务 ，例如线上教育 、线上娱乐 、线上购物等 ，在进行线',
                  '下消费时 ，更多考虑实体店的通风性 、安全性、是否进行消毒 、是',
                  '否人员聚集等条件 ；其次，共享消费模式会受到一定程度的影',
                  '响，消费者会考虑共享汽车 、共享单车等卫生程度 ，减少人员接',
                  '触，倾向于“非接触式消费 ”，这会促进远程消费模式发展 ；最后，',
                  '居民对于健康安全方面消费增多 ，由于新冠肺炎疫情 ，居民意识',
                  '到身体素质 的重要性 ，更加注意安全防护 ，对于保健品 、防护用',
                  '品、消毒用具 、药品等消费增多 ，对于医疗保健 、健康运动 、保险', '等服务性产品消费也相应增多 。',
                  '3.海外消费回流国内', '由于中国政府的强力举措 ，全国人民上下一心 ，众志成城 ，',
                  '广大医护人员 、志愿者、服务人员投入抗疫工作中 ，国内疫情得',
                  '以控制住 ，并很大程度缓解 ，但是境外疫情现在到达了爆发期 ，',
                  '且没有得到很好的控制 ，目前美国新冠肺炎确诊人数现已超 过',
                  '100万，国内现在严防境外输入病例 ，许多中外合资院校已经 确',
                  '定本学期不返校 ，全球范围内人员流动受到影响 ，居民在韩 国、',
                  '日本、美国、德国等国家的海外消费受到很大影响 ，加上国外 经',
                  '济受到疫情影响 ，而中国开始积极推进复工复产 ，消费逐步回 流', '国内。',
                  '五、企业对消费者行为模式变化后的应对措施', '1.预测消费变化趋势 ，挖掘新的商机',
                  '未受到较大影响导致停工停产的企业 ，在应对新冠疫情导',
                  '致的消费者行为模式变化后 ，应该积极预测消费变化趋势 ，做好',
                  '安全防控工作并履行社会责任的前提下 ，挖掘新的商机 ，关于居',
                  '民对于安全 、健康、便捷的消费诉求快速响应 ，采取一系列措施',
                  '加以满足 。并且根据未来预测的变化趋势 ，企业调整下一步发展', '目标以及计划 。',
                  '2.融合新的营销渠道 ，创新经营模式', '居民消费呈现线上线下消费融合趋势 ，企业应该根据这一',
                  '消费模式变化 ，进行营销升级 ，融合新的营销渠道 ，实现线上线',
                  '下一体化 ，一方面企业办公可以采取线上线下结合模式 ，提高员',
                  '工办公效率 ，方便未能到岗员工创造价值 ，另一方面企业为客户',
                  '群体提供产品服务也可以开通线上线下融合模式 ，抓住客户群',
                  '体需求，获取更大利益 。同时企业也可以抓住市场整合机遇 ，优', '化资源配置 ，更好地满足消费者的需求变化 。',
                  '3.抓住消费风口 ，拓展市场领域', '企业应该抓住新冠肺炎疫情这一契机 ，了解消费行为改变',
                  '后新的市场风口 ，拓展市场领域 ，重新定位新的目标客户群体以',
                  '及新的非接触式消费需求 ，并提出针对性发展举措 。企业应该牢',
                  '牢把握住疫情影响对医药 、教育、娱乐等产业来带的新的机遇 ，开辟新的发展领域 ，并制定可持续发展策略 。',
                  '六、推动国内市场发展建议', '1.加强健康消费教育宣传引导',
                  '居民消费是国内市场发展的主要推动力 ，在疫情影响下的',
                  '经济寒冬 ，国家政府也应该采取一系列举措推动国内市场发展 ，',
                  '为企业以及经营个体户创造良好的发展环境 。从青少年抓起 ，加',
                  '强健康教育宣传引导工作 ，帮助他们确立正确的消费观念 ，以及',
                  '倡导良好的公共卫生习惯 ，通过播放宣传片 、公益广告 ，开展志',
                  '愿活动等形式 ，做好健康普及工作 ，进一步引导全体居民健康消',
                  '费的观念意识 ，战胜疫情 ，推动居民健康消费 ，营造良好的国内', '市场发展氛围 。', '2.增加居民收入保障',
                  '面对天灾 ，政府应该积极给予保障措施用来保护居民权益 ，',
                  '例如给无法及时复工复产的居家人员一定的生活保障 ，或是财',
                  '政拨款保障居民养老保险 、医疗保险 ，以及完善基础设施建设 ，',
                  '加强失业保障 、提供就业援助 ，可以稳定居民消费水平 。另外鼓',
                  '励社会资本参与教育 、医疗、文化设施建设 ，降低居民税收负担', '鼓励消费 。', '3.提升商品服务质量监管',
                  '疫情期间出现了一些故意囤货 、哄抬物价的行为 ，更有出现',
                  '制造假冒伪劣商品 、以次充好 ，赚取不正当利益的欺诈行为 。对',
                  '此政府应该加强监管 ，严厉制止这种行为 ，保证物价水平不乱',
                  '涨，稳定居民消费 ，另一方面对于商品服务质量也应该有完善的', '监督举措 ，对非法行为依法惩处 。',
                  '4.正确引导舆论发展方向', '新冠肺炎疫情期间 ，社交媒体平台上出现了一些别具用心',
                  '的人散布不实谣言 ，引得人人自危 ，相关部门应该加以监管 ，做',
                  '到信息公开化 、透明化，严厉打击谣言散布者 ，并加以宣传 ，让居',
                  '民不信谣 、不传谣，正确引导舆论发展方向 ，有助于维护国内市',
                  '场稳定发展 ，维持物价稳定 ，保证居民正常 、健康消费 。',
                  '2020年到来的新型冠状肺炎疫情是对全人类的一大重要考',
                  '验，国家在面对紧急突发公共卫生事件 ，每个人紧紧团结在一',
                  '起，以坚决的姿态走出 “逆行者”的风采，政府也采取了一系列 强',
                  '有力的举措 ，控制了疫情发展 ，并加快推进复工复产 、返工返学，',
                  '但是战役并未结束 ，我们不能掉以轻心 ，更应该积极采取举 措恢',
                  '复经济发展 ，维护市场稳定 ，努力全面建成小康社会和实现 2020', '年经济社会发展目标 。', '参考文献 ：',
                  '[1]关利欣 ，梁威.中美消费发展升级历程比较及启示 [J].中国流通经济 ，', '2019(5):13-21.',
                  '[2]郑江淮 ，付一夫 ，陶金.新冠肺炎疫情对消费经济的影响及对策分析', '[J].消费经济 ，2020,(2):3-9.',
                  '[3]梁威，关利欣 ，胡雪.消费国际化趋势下的中国对策 [J].国际贸易 ，', '2020(2):25-29.',
                  '[4]Keynes,J. M. The General Theory of Employment,In terest and',
                  'Money[M].United Kingdom:Palgrave Macmillan, 1936.',
                  '[5]欧阳金雨 .及早抓住疫情过后的消费机会 [N].湖南日报 ,2020', '-02-23(003).',
                  '作者简介 ：谭诗怡（1999.10- ），女，籍贯:湖南湘乡 ，湘潭大学，本', '科在读，研究方向 :电子商务',
                  '11Copyright©博看网 www.bookan.com.cn. All Rights Reserved.']
    paragraphs = ['Hello!\nHi!\nGoodbye!']
    tc = TextChunker()
    chunk_size = 512
    chunks = tc.chunk_sentences(paragraphs, chunk_size)

    for i, chunk in enumerate(chunks):
        print(f"块 {i + 1}: {chunk}\n")
