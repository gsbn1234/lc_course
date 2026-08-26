"""
生成 RAG 知识库的 PDF 文档集（8 个主题）。
覆盖向量数据库、嵌入模型、Transformer、LangChain、RAG 高级技术、
Prompt Engineering、AI Agent、LLM 基础知识。
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 中文字体注册 ==========
FONT_PATHS = [
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

cn_font = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("CN", fp))
            cn_font = "CN"
            break
        except Exception:
            continue

if cn_font is None:
    cn_font = "Helvetica"

# ========== 样式 ==========
styles = getSampleStyleSheet()
title_style = ParagraphStyle("T", fontName=cn_font, fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=20)
h1_style = ParagraphStyle("H1", fontName=cn_font, fontSize=16, leading=22, spaceBefore=18, spaceAfter=10)
h2_style = ParagraphStyle("H2", fontName=cn_font, fontSize=13, leading=18, spaceBefore=12, spaceAfter=6)
body_style = ParagraphStyle("B", fontName=cn_font, fontSize=10, leading=16, spaceBefore=2, spaceAfter=6)
code_style = ParagraphStyle("C", fontName=cn_font, fontSize=9, leading=14, backColor="#F5F5F5", leftIndent=20, spaceBefore=4, spaceAfter=4)

def P(text, style=body_style):
    return Paragraph(text, style)

def S(h=12):
    return Spacer(1, h)

def make_pdf(filename, title, author, category, difficulty, date, sections):
    path = os.path.join(OUTPUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.5*cm, rightMargin=2.5*cm)
    doc.title = title
    doc.author = author
    doc.subject = category

    story = []
    story.append(P(title, title_style))
    story.append(S(6))
    info = f'分类: {category}  |  难度: {difficulty}  |  日期: {date}  |  作者: {author}'
    story.append(P(info, body_style))
    story.append(S(12))

    for sec_title, paragraphs in sections:
        story.append(P(sec_title, h1_style))
        for para in paragraphs:
            if para.startswith('///code'):
                story.append(P(para[7:], code_style))
            else:
                story.append(P(para, body_style))
        story.append(S(6))

    doc.build(story)
    print(f'  OK {filename} ({len(sections)} chapters)')



# ================================================================
# PDF 1: 向量数据库
# ================================================================
make_pdf(
    'vector_databases.pdf',
    '向量数据库深度对比: FAISS、Milvus、Chroma 与 Pinecone',
    '向量数据库研究组',
    '向量数据库',
    '中级',
    '2026-06-15',
    [
        ('第一章: 向量数据库概述', [
            '向量数据库是一种专门用于存储、索引和检索高维向量数据的数据库系统。在大语言模型时代，向量数据库成为 RAG(检索增强生成)系统的核心基础设施。',
            '向量数据库的核心能力包括: 高效存储高维向量(通常 384-4096 维)、支持近似最近邻搜索(ANN)、提供元数据过滤、支持分布式部署和数据持久化。',
            '与传统的结构化数据库(如 MySQL)或全文搜索引擎(如 Elasticsearch)不同，向量数据库的查询基于语义相似度而非精确匹配。这使得它能够处理近义词、同义改写等传统检索难以应对的场景。',
        ]),
        ('第二章: FAISS -- 本地高性能向量检索', [
            'FAISS(Facebook AI Similarity Search)是 Meta 开源的高性能向量相似性搜索库。它最早发布于 2017 年，目前已经成为学术界和工业界最广泛使用的向量检索工具之一。',
            'FAISS 的核心优势在于: 极致的检索性能--支持 GPU 加速和数十亿级别的向量检索；多种索引结构可选--FlatL2、IVF、HNSW、PQ 等，覆盖了精度与速度的不同取舍；零部署成本--纯本地运行，不需要额外的基础设施。',
            'FAISS 的局限同样明显: 本身不是数据库--没有持久化机制，需要额外实现；单机运行--不支持分布式部署；没有元数据过滤--如果需要按时间、类别等字段筛选，需要在 FAISS 之上自己实现。',
            'FAISS 的 IVF(Inverted File)索引是最常用的加速结构。IVF 先用 K-means 聚类将向量空间划分为多个单元，查询时只搜索最近的几个单元(nprobe 参数控制)，大幅降低计算量。IVF 配合 PQ(Product Quantization)压缩，可以在几乎不损失精度的情况下将内存占用降低到原来的 1/10。',
            '///codeFAISS 使用示例(Python):\nimport faiss\nimport numpy as np\n\n# 创建 128 维向量索引\ndimension = 128\nindex = faiss.IndexFlatL2(dimension)\n\n# 添加向量\nvectors = np.random.random((10000, dimension)).astype("float32")\nindex.add(vectors)\n\n# 搜索 top-5\nquery = np.random.random((1, dimension)).astype("float32")\nD, I = index.search(query, 5)',
        ]),
        ('第三章: Milvus -- 云原生分布式向量数据库', [
            'Milvus 是 LF AI & Data 基金会旗下的开源向量数据库，专为云原生场景设计。它提供了完整的 CRUD 操作、数据持久化、分布式水平扩展、元数据过滤和混合搜索能力。',
            'Milvus 的架构采用了存储计算分离的设计理念。系统由接入层(Proxy)、协调层(Root/Query/Data Coordinators)、工作层(Query/Data Nodes)和存储层(对象存储 + 消息队列)四层组成。这种架构使得各层可以独立扩缩容。',
            '在索引方面，Milvus 支持 IVF_FLAT、IVF_SQ8、IVF_PQ、HNSW、ANNOY、DISKANN 等十余种索引类型。对于十亿级数据集，推荐使用 IVF_PQ 配合 GPU 加速；对于千万级以内且追求极致精度的场景，HNSW 是最佳选择。',
            'Milvus 的元数据过滤是其区别于 FAISS 的核心功能之一。用户可以在检索时通过标量字段(如时间范围、文档类型、状态等)预过滤候选集，然后再做向量相似度检索。这种「标量过滤 + 向量检索」的组合就是 Self Query Retriever 的底层原理。',
        ]),
        ('第四章: Chroma -- 轻量级 AI 原生向量数据库', [
            'Chroma 是为 AI 应用而生的轻量级向量数据库。它的设计哲学是简单优先--几行代码即可完成从文档加载到向量存储的全流程，非常适合原型开发和小规模项目。',
            'Chroma 的一个特色功能是内置了 Embedding 支持。用户可以直接将一个文本字符串传入 Chroma，它内部会自动调用嵌入模型(如 OpenAI Embeddings、SentenceTransformers 等)将文本向量化后存储。这意味着开发者不需要自己管理嵌入管道。',
            '在部署方面，Chroma 支持两种模式: 嵌入式模式(进程内运行，适合单机开发)和客户端-服务器模式(支持远程访问，适合团队协作)。对于学习 RAG 的开发者而言，Chroma 的嵌入式模式配合 LangChain 是最简单的入门组合。',
        ]),
        ('第五章: Pinecone -- 全托管云端向量数据库', [
            'Pinecone 是一个全托管的云端向量数据库服务。它最大的卖点是零运维--不需要管理任何基础设施，不需要做索引调优，不需要担心数据备份。',
            'Pinecone 的索引在创建后会自动进行性能优化，包括索引分片、查询路由和负载均衡。它的默认配置适用于大多数场景，但高级用户也可以自定义 pod 类型、副本数、索引类型等参数。',
            'Pinecone 的价格模型基于 Pod 规格而非数据量或查询次数，适合需要稳定成本的商业应用。但对于个人开发者和学习用途而言，FAISS + Chroma 的免费方案通常已经足够。',
        ]),
        ('第六章: 选型建议', [
            '对于 RAG 初学者: Chroma(嵌入式模式)+ LangChain 是最简单的组合，5 分钟即可搭好原型。',
            '对于本地开发和小规模生产: FAISS 配合 HNSW 索引，结合自建的元数据过滤层。你的 lc_course 项目已经在实践这一方案。',
            '对于需要持久化和分布式部署的生产系统: Milvus 提供了完整的数据库能力，包括数据持久化、水平扩展、混合搜索和丰富的监控指标。',
            '对于需要零运维体验的团队: Pinecone 可以让开发者专注于应用逻辑而非基础设施，但需要接受厂商锁定和较高的使用成本。',
        ]),
    ]
)

# ================================================================
# PDF 2: 嵌入模型
# ================================================================
make_pdf(
    'embedding_models.pdf',
    '文本嵌入模型深度解析: 从 Word2Vec 到 BGE-M3',
    '嵌入模型实验室',
    '嵌入模型',
    '中级',
    '2026-06-20',
    [
        ('第一章: 什么是文本嵌入', [
            '文本嵌入(Text Embedding)是将非结构化的文本数据转换为固定维度的稠密向量的技术。这个向量--通常是 384、768 或 1024 维的浮点数数组--是文本在高维语义空间中的坐标表示。',
            '嵌入的核心特性是语义相近的文本在向量空间中距离更近。例如，[什么是 RAG] 和 [检索增强生成的定义] 的嵌入向量之间的余弦相似度会远高于 [什么是 RAG] 和 [今天天气不错] 之间的相似度。',
            '在 RAG 系统中，嵌入模型负责两件事: 索引阶段--将知识库中的所有文档块转换为向量并存入向量数据库；查询阶段--将用户问题转换为向量，与库中的文档向量计算相似度，找到最相关的文档。嵌入模型的质量直接决定了 RAG 系统的检索质量。',
        ]),
        ('第二章: BGE 系列模型', [
            'BGE(BAAI General Embedding)是北京智源人工智能研究院(BAAI)发布的一系列中文嵌入式模型。BGE 系列包括 bge-small-zh、bge-base-zh、bge-large-zh 三个中文版本。',
            'bge-small-zh-v1.5 是目前最受欢迎的中文嵌入模型之一。它仅有 24 层、1024 隐藏维度、总参数量约 2400 万，输出 512 维向量。尽管体积小巧，它在 MTEB 中文榜单上的表现超越了多个参数量更大的模型。',
            'bge-large-zh-v1.5 提供了更高的精度，输出 1024 维向量，但参数量也增加到约 3.26 亿。在需要极高检索精度的场景(如法律文书检索、医疗文献搜索)中，large 版本的优势会更加明显。',
            'BGE-M3 是 BAAI 在 2024 年发布的多语言通用嵌入模型。它的 M3 代表三个特性: Multi-Lingual(支持 100+ 语言)、Multi-Functionality(同时支持稠密检索和稀疏检索)、Multi-Granularity(支持不同长度的输入，从短词到 8192 token 的长文档)。',
            '在你的 lc_course 项目中使用的 bge-small-zh-v1.5 是一个非常合理的选择--对于小规模知识库，small 版本的速度优势远大于大版本可能带来的精度提升。当知识库扩展到数万页时，考虑升级到 bge-large 或 BGE-M3。',
        ]),
        ('第三章: 中文嵌入模型生态', [
            'text2vec-large-chinese 是由 shibing624 发布的中文 Sentence-BERT 模型。它基于 BERT-base 架构，使用 CoSENT(Cosine Sentence)损失函数训练，在中文语义相似度任务上表现优异。输出 768 维向量。',
            'm3e(Moka Massive Mixed Embedding)是由 Moka AI 训练的中文嵌入模型系列，包括 m3e-small、m3e-base、m3e-large。m3e 系列在 2200 万+ 的中文句对数据集上训练，覆盖了社区问答、百科知识、新闻等多种场景。',
            'OpenAI 的 text-embedding-3-small 和 text-embedding-3-large 是目前最强的商业嵌入模型之一。它们支持最高 3072 维输出，并且可以通过传递 dimensions 参数灵活控制输出维度而不会显著损失精度。缺点是按 token 收费，在大规模场景下成本较高。',
        ]),
        ('第四章: 嵌入模型的评估指标', [
            'MTEB(Massive Text Embedding Benchmark)是目前最权威的嵌入模型评估基准。它覆盖了分类、聚类、对匹配、重排序、检索和摘要等 7 大类共 58 个数据集。',
            '对于 RAG 系统，最关键的 MTEB 子指标是 Retrieval(检索)。检索评测通常使用 NDCG@10、MAP@10 和 Recall@100 作为标准指标。NDCG@10 衡量前 10 个检索结果的排序质量，Recall@100 衡量在 100 个候选结果中能否覆盖正确答案。',
            '在选择嵌入模型时，不要只看 MTEB 总分--要根据自己的场景看对应的子指标。如果你的 RAG 系统主要处理中文技术文档，应该重点关注 MTEB 中文检索子榜单上的表现。',
        ]),
        ('第五章: 嵌入模型的微调', [
            '通用嵌入模型在垂直领域(如医疗、法律、金融)的表现往往不如预期。这是因为这些领域的术语和专业表达在通用训练语料中出现频率低，模型无法准确理解其语义。',
            '领域微调可以显著提升嵌入模型在特定领域的表现。常用的微调方法包括: 使用领域内的问答对作为正例、使用对比学习(Contrastive Learning)优化正负例距离、使用知识蒸馏将大模型的知识迁移到小模型。',
            '一个实践建议: 对于 RAG 初学者，先用通用嵌入模型(如 bge-small-zh)搭建端到端的管线，等到整个管线跑通并且评估分数稳定后，再考虑微调。过早优化嵌入层而忽视检索策略、重排序等其他环节，往往是时间上的浪费。',
        ]),
    ]
)

# ================================================================
# PDF 3: Transformer 架构
# ================================================================
make_pdf(
    'transformer_architecture.pdf',
    'Transformer 架构入门: 注意力机制详解',
    '深度学习学院',
    '大语言模型',
    '初级',
    '2026-05-10',
    [
        ('第一章: 为什么需要 Transformer', [
            '在 Transformer 出现之前，自然语言处理领域的主流架构是 RNN(循环神经网络)和 LSTM(长短期记忆网络)。这些模型按序处理文本--第 3 个词必须等第 2 个词算完之后才能开始计算。这种串行方式带来了两个问题: 训练速度慢(无法并行化)和长距离依赖丢失(句子后半段的词很难和前半段建立联系)。',
            '2017 年，Google 的研究团队在论文《Attention Is All You Need》中提出了 Transformer 架构。它的核心创新是自注意力机制(Self-Attention)--句子中的每个词在编码时，都可以直接看到句子中的所有其他词，然后根据语义相关性分配不同的注意力权重。这使得模型可以并行处理整个序列，同时捕捉任意距离的依赖关系。',
        ]),
        ('第二章: 自注意力机制(Self-Attention)', [
            '自注意力是 Transformer 的灵魂。它的计算过程可以概括为三个步骤: 首先，将每个输入词通过三个不同的权重矩阵投影到查询(Q)、键(K)、值(V)三个空间；然后，计算每个词的 Q 与所有词的 K 的点积，得到一个注意力分数矩阵；最后，用 Softmax 将分数归一化为权重，加权求和所有的 V 得到输出。',
            '用通俗的方式理解自注意力: 想象你在读一句话--[猫追老鼠，它跑得很快]。当你读到[它]这个字时，你需要判断它指的是猫还是老鼠。自注意力机制就是让模型自动计算「它」与句中每个词的相关性分数，结果发现「它」和「老鼠」的关联度最高，于是将更多的注意力放在「老鼠」上。',
            '数学上，自注意力的公式为: Attention(Q, K, V) = softmax(Q x K^T / sqrt(d_k)) x V。其中 d_k 是 Key 向量的维度，除以 sqrt(d_k) 是为了防止点积过大导致 Softmax 梯度消失。',
            '///code自注意力的简化 Python 实现:\nimport torch\nimport torch.nn.functional as F\n\ndef self_attention(Q, K, V):\n    d_k = K.size(-1)\n    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)\n    weights = F.softmax(scores, dim=-1)\n    return weights @ V',
        ]),
        ('第三章: 多头注意力(Multi-Head Attention)', [
            '单头自注意力只能从一种角度去理解词与词之间的关系。多头注意力并行运行多个独立的自注意力头，每个头从不同的角度(语法、语义、位置、指代等)捕捉不同的关系模式。',
            '例如，8 头注意力在编码[猫追老鼠，它跑得很快]时: 第 1 个头可能专注于语法的主谓宾结构；第 3 个头可能专注于指代消解；第 5 个头可能关注修饰关系。所有头的输出拼接后经过一个线性变换，得到最终的融合表示。',
            'Transformer 原始论文使用了 8 个注意力头(h=8)、每个头的维度 d_k = d_v = 64(总维度 d_model = 512)。后来的大模型如 GPT-3 使用了 96 个头(d_model = 12288)。头数越多，模型能同时关注的角度越多，但计算开销也越大。',
        ]),
        ('第四章: 位置编码(Positional Encoding)', [
            '自注意力机制的一个固有缺陷是无法感知词的位置--因为它在计算时平等地看待所有位置。如果没有位置信息，[我爱你]和[你爱我]在 Transformer 眼里是一模一样的。',
            '位置编码就是给每个位置注入唯一的位置信号。原始 Transformer 使用正弦/余弦函数生成位置编码。这种编码的好处是固定的、不需要训练、且可以外推到训练时未见过的序列长度。',
            '后来的模型(如 GPT、BERT)更多地使用可学习的位置编码(Learned Positional Encoding)--位置编码和词嵌入一起作为模型参数参与训练。RoPE(Rotary Position Embedding，旋转位置编码)是近年来最流行的方案，它通过旋转矩阵将位置信息直接融入注意力计算中，被 LLaMA、Qwen、ChatGLM 等主流模型采用。',
        ]),
        ('第五章: 前馈网络与残差连接', [
            '每个 Transformer 块除了自注意力层之外，还包含一个前馈网络(Feed-Forward Network，FFN)和两个残差连接(Residual Connection)。',
            'FFN 是一个简单的两层全连接网络: FFN(x) = ReLU(xW1 + b1)W2 + b2。它的作用是给每个位置独立地增加非线性变换能力。和自注意力层不同，FFN 层中不同位置之间没有交互--每个位置的变换是独立的。',
            '残差连接的作用是让梯度能直接跳过层反向传播，避免深层网络的梯度消失问题。归一化层(Layer Normalization)则保证了每层的输出分布稳定。这三者的组合--Attention + FFN + Residual + LayerNorm--构成了一个完整的 Transformer Block，GPT-3 堆叠了 96 个这样的 Block。',
        ]),
    ]
)

# ================================================================
# PDF 4: LangChain 高级用法
# ================================================================
make_pdf(
    'langchain_advanced.pdf',
    'LangChain 高级编程: Chain、Agent、Memory 与工具调用',
    'LangChain 开发组',
    '开发框架',
    '高级',
    '2026-07-05',
    [
        ('第一章: LCEL -- LangChain 表达式语言', [
            'LCEL(LangChain Expression Language)是 LangChain 最核心的编程范式。它的语法非常简单: 用竖线(|)将组件串联。例如 prompt | llm | output_parser 表示将 prompt 的输出传给 llm，再将 llm 的输出传给 output_parser。',
            'LCEL 的本质是实现了 Runnable 协议。每一个 LangChain 组件(PromptTemplate、ChatModel、Retriever、Tool)都是一个 Runnable，这意味着它们都可以用 | 串联、用 .invoke() 调用、用 .batch() 批量处理、用 .stream() 流式输出。',
            'LCEL 的一个重要特性是自动并行化。当一个 Runnable 有多个输入来源且互不依赖时，LangChain 会自动将它们并行执行。例如 RunnableParallel 可以同时运行总结链和翻译链。',
            '///codeLCEL 示例:\nfrom langchain_core.prompts import ChatPromptTemplate\nfrom langchain_core.output_parsers import StrOutputParser\n\nprompt = ChatPromptTemplate.from_template("翻译: {text}")\nchain = prompt | llm | StrOutputParser()\nresult = chain.invoke({"text": "Hello World"})',
        ]),
        ('第二章: Chain -- 可组合的处理链', [
            'Chain 是 LangChain 中最基础的工作流抽象。最简单的 Chain 是 LLMChain--一个 prompt + 一个 LLM，输入变量、输出文本。复杂的 Chain 可以嵌套多个子 Chain，形成多层级的处理流程。',
            'SequentialChain 将多个 Chain 按顺序连接，前一个 Chain 的输出成为后一个 Chain 的输入。例如: 文档总结链 -> 关键信息提取链 -> 格式化输出链，三个步骤顺序执行。',
            'RunnableParallel 允许多个 Chain 并行执行。在 RAG 场景中，你可以同时运行 vector_search 和 bm25_search 然后合并结果，这在你的 parent_hybrid_retrieve 中已经手动实现了。用 LCEL 的 RunnableParallel 可以让代码更简洁。',
        ]),
        ('第三章: Memory -- 对话记忆管理', [
            'Memory 是 LangChain 中管理对话历史和上下文的组件。在单轮 RAG 中不需要 Memory，但在多轮对话场景中，模型需要记住之前说过什么才能自然地追问。',
            'ConversationBufferMemory 是最简单的实现--它将完整的对话历史保存在一个列表中，每轮对话时将全部历史拼入 Prompt。简单但昂贵: 随着对话增长，token 消耗线性增加。',
            'ConversationSummaryMemory 在每轮对话后自动用 LLM 摘要历史，只保留摘要而非完整历史。这使得 token 消耗保持稳定，但可能丢失细节信息。在需要精确回忆细节的场景(如客服工单处理)中，摘要方案不够可靠。',
            '你的项目中目前没有引入多轮对话机制--graph.invoke() 每次都是独立的单轮调用。如果后续要支持追问，需要在 AgenticRAGState 中增加 chat_history 字段，然后在 generate_node 中将历史对话一并传入。',
        ]),
        ('第四章: Tool Calling -- 让 LLM 调用外部工具', [
            'Tool Calling (工具调用)是 LLM 从语言模型进化为行动引擎的关键能力。通过 Tool Calling，LLM 可以调用搜索引擎、数据库查询、代码执行器、API 接口等外部工具来获取它自身不具备的信息或能力。',
            '在 LangChain 中，工具被封装为 @tool 装饰器或 StructuredTool 对象。每个工具定义了自己的名称、描述(告诉 LLM 这个工具是干什么的)、参数 schema(告诉 LLM 调用时应该传什么参数)。',
            'LLM 在 Tool Calling 时并不实际执行工具--它只生成一个工具调用请求(包含工具名称和参数)。框架收到这个请求后，执行对应的工具函数，将结果返回给 LLM，LLM 再基于结果生成最终回答。这个决策-执行-观察的循环就是 Agent 的核心工作模式。',
        ]),
        ('第五章: 输出解析器与结构化输出', [
            'LLM 的默认输出是自由文本。但在很多场景中，我们需要结构化的输出--比如提取 JSON 格式的实体关系、生成 Markdown 表格、返回枚举值。输出解析器(Output Parser)就是做这件事的。',
            'StrOutputParser 是最简单的解析器--直接返回原始文本。JsonOutputParser 让 LLM 输出严格的 JSON 格式，并通过 Pydantic schema 验证字段类型。在 Self Query Retriever 中，LangChain 内部就是用一个专门的输出解析器来提取 LLM 生成的过滤条件。',
            '///code结构化输出示例:\nfrom langchain_core.output_parsers import PydanticOutputParser\nfrom pydantic import BaseModel, Field\n\nclass PersonInfo(BaseModel):\n    name: str = Field(description="人物姓名")\n    age: int = Field(description="年龄")\n    profession: str = Field(description="职业")\n\nparser = PydanticOutputParser(pydantic_object=PersonInfo)',
        ]),
    ]
)

# ================================================================
# PDF 5: RAG 高级技术
# ================================================================
make_pdf(
    'rag_advanced.pdf',
    'RAG 高级技术: 从基础检索到自主决策',
    'RAG 研究组',
    'RAG',
    '高级',
    '2026-07-18',
    [
        ('第一章: 基础 RAG 的局限性', [
            '标准的 RAG 管线虽然强大，但在实际应用中有几个明显的局限。第一，固定检索--每次只搜一轮，不管结果好不好；第二，无差别对待--所有问题走相同的检索策略；第三，缺少结构化理解--无法利用文档的元数据做精确过滤。',
            '这些局限在实际场景中会导致具体的问题。例如用户问「第三章里提到的优化方法有哪些」--基础 RAG 没有章节的概念，会全库搜索优化方法，返回的可能是第五章的内容。又如用户问「2024 年之后的版本有什么变化」--基础 RAG 不理解时间过滤。',
            '业界针对这些局限发展出了多条改进路线: Self Query(结构化过滤)、Query Decomposition(问题拆解)、Respondent RAG(自适应检索)、和 Agentic RAG(自主决策回路)。下面各章逐一展开。',
        ]),
        ('第二章: Self Query Retriever -- 结构化过滤检索', [
            'Self Query Retriever 的核心能力是从用户的自然语言问题中自动抽取结构化过滤条件。例如用户问「2024 年之后发布的中级难度的向量数据库文档」，Self Query 会将查询拆解为两部分: 语义查询--「向量数据库文档」，用于向量相似度搜索；结构化过滤--{date: >2024, difficulty: 中级}，用于缩小检索范围。',
            'LangChain 的 SelfQueryRetriever 实现方式: 首先定义元数据字段的 schema(哪些字段、什么类型、可选值范围)，然后 LLM 从用户问题中提取过滤条件并输出结构化 JSON，框架将过滤条件应用到向量数据库的检索中。整个过程对用户透明--用户依然用自然语言提问，但检索变得更精准。',
            'Self Query 和普通过滤的区别在于灵活性。普通过滤需要开发者手动构造 filter 字典，Self Query 由 LLM 自动生成。这就是你接下来要学的内容--agentic_router_graph.py 的下一步，不需要用户显式选择过滤条件，系统自己判断。',
            '///codeSelfQueryRetriever schema 定义示例:\nfrom langchain.chains.query_constructor.base import AttributeInfo\n\nmetadata_field_info = [\n    AttributeInfo(name="category", description="文档分类", type="string"),\n    AttributeInfo(name="difficulty", description="难度等级", type="string"),\n    AttributeInfo(name="date", description="发布日期", type="date"),\n    AttributeInfo(name="source", description="来源PDF文件名", type="string"),\n]',
        ]),
        ('第三章: Query Decomposition -- 复杂问题拆解', [
            'Query Decomposition 解决的是复杂多步推理问题。例如用户问「FAISS 和 Milvus 在处理十亿级向量时，各有什么优劣？」--这是一个典型的对比分析问题，单次检索很难召回两边各自的优缺点。',
            'Query Decomposition 的策略是: 用 LLM 将复杂问题拆解为多个子问题，每个子问题单独检索，最后将所有检索结果汇总。拆解后的子问题可能是: 「FAISS 十亿级向量检索的索引方案和性能」、「Milvus 十亿级向量检索的索引方案和性能」。',
            '在你的项目中，multi_query.py 做的事情是同一意图的不同角度--3 条查询围绕同一个点。而 Query Decomposition 是不同子主题的独立检索--每条子问题搜不同的东西。两者配合使用效果更好: 先拆解，再对每个子问题做 multi_query。',
        ]),
        ('第四章: HyDE -- 假想文档嵌入的深入分析', [
            'HyDE(Hypothetical Document Embeddings)的原理已经在你的 hyde.py 中实现。这里补充一些进阶细节。',
            'HyDE 的一个关键问题是: 假想答案的「幻觉」程度会影响检索质量吗？研究表明，事实错误的假想答案仍然有效--因为检索依赖的是语义方向而非事实准确性。但风格偏差的假想答案(如过于口语化、没有技术密度)的检索效果会显著下降。',
            '解决风格偏差的方法之一是在 HyDE Prompt 中明确指定文档风格: 「请用技术白皮书的口吻写一段回答」。你的 hyde.py 中已经这样做了，这是正确且重要的。',
            '另一个进阶技巧是 HyDE + 多样性采样: 生成 3 个不同风格的假想答案(技术型、科普型、实操型)，对每个假想答案分别检索，然后合并排序。这在你的评估结果中能得到验证--技术概念题的 Recall 最好，简单入门题的 Relevancy 有时反而下降，说明假想答案的风格应该和问题难度匹配。',
        ]),
        ('第五章: Agentic RAG -- 自主决策的检索系统', [
            'Agentic RAG 的核心思想是: 让 LLM 成为检索策略的决策者，而不仅仅是最终答案的生成者。传统的 RAG 是一个固定流水线，Agentic RAG 是一个自主的决策回路。',
            '你在 agentic_router_graph.py 中实现的 reflect_node 就是 Agentic RAG 的最小可行版本: LLM 审视检索结果 -> 判断是否够用 -> 够就生成答案、不够就换个角度再搜。这个简单的反思回路已经让你的管线从固定工序升级为自主决策。',
            '更完整的 Agentic RAG 还可以加入: 策略选择(直接搜 vs HyDE vs 拆解后分别搜)、工具切换(本地搜不到就联网搜)、来源融合(PDF 来源 + Web 来源，分别标注可信度)、和迭代深化(第一轮搜到概述 -> 发现需要更细节的信息 -> 第二轮精准搜索)。',
        ]),
    ]
)

# ================================================================
# PDF 6: Prompt Engineering
# ================================================================
make_pdf(
    'prompt_engineering.pdf',
    'Prompt Engineering 系统指南: 从入门到精通',
    'Prompt 工程组',
    'Prompt Engineering',
    '初级',
    '2026-04-22',
    [
        ('第一章: 什么是 Prompt Engineering', [
            'Prompt Engineering(提示词工程)是设计和优化输入给大语言模型的提示词，以引导模型产出高质量、符合预期的输出的系统性方法。它不是一次性写一句话，而是一个迭代的、可以量化的工程过程。',
            '一个好的 Prompt 通常包含四个要素: 角色设定(告诉模型它是谁)、任务描述(告诉模型要做什么)、输出格式(告诉模型怎么呈现结果)、示例(用例子教会模型期望的输出模式)。这四个要素并非每次都需要，但对于复杂任务，缺一不可。',
            '以你的 prompt.py 为例: 「你是一个严谨的知识问答助手」就是角色设定，「请仅根据参考资料回答问题」是任务描述，「如果资料中没有答案，请明确说明」是处理边缘情况的约束。这是一个简洁有效的 RAG Prompt。',
        ]),
        ('第二章: Few-Shot Prompting -- 用示例引导输出', [
            'Few-Shot Prompting 是在 Prompt 中提供少量示例来引导 LLM 理解输出格式和风格的技术。例如，如果你想让 LLM 将文本分类为正面或负面，提供 3-5 个标注好的样本比单纯描述规则更有效。',
            '示例的选择原则: 多样性(覆盖不同的输入模式和输出类型)、代表性(使用真实场景中可能出现的边界情况)、顺序(关键示例放在最后，因为 LLM 对 Prompt 末尾的信息更敏感--这被称为 Recency Bias)。',
            '在你的评估脚本中，三个评分 Prompt 都使用了「例如: 0.75」这种微型 Few-Shot 来引导 LLM 输出纯数字而非解释文字。这是一个很好的实践--只用了一个示例就大幅提高了输出格式的准确率。',
        ]),
        ('第三章: Chain-of-Thought -- 让模型展示推理过程', [
            'Chain-of-Thought (CoT，思维链)提示是一种让 LLM 在给出最终答案前先展示中间推理步骤的技术。研究表明，带有 CoT 的 Prompt 能在数学推理、逻辑推断、多步问答等复杂任务上提升 10-30% 的准确率。',
            'CoT 的工作原理是: LLM 是自回归模型(逐 token 生成)，前面的 token 会影响后面 token 的概率分布。如果让模型先生成「让我一步步思考...」，它就更可能进入推理模式；如果让它直接输出答案，它可能跳过中间步骤导致错误。',
            '在你的 RAG 管线中，CoT 的一个应用场景是在 generate_node 的 Prompt 中加入「请先列出上下文中与问题相关的关键事实，再基于这些事实回答」。这会让模型在生成答案前先做一轮内部分析，提高忠实度。',
        ]),
        ('第四章: ReAct -- 推理与行动交替', [
            'ReAct(Reasoning + Acting)是 Agentic RAG 的理论基础。它的核心模式是: Thought(思考) -> Action(行动) -> Observation(观察) -> Thought -> Action -> ... -> Final Answer。',
            '在你的 agentic_router_graph.py 中，reflect_node 本质上就是一个简化的 ReAct 循环: Thought(审视上下文是否有缺口) -> Action(生成新搜索词或不生成) -> Observation(新检索结果) -> Thought(再次审视)。不同的是，ReAct 通常把思考过程显式输出给用户看，而你的 reflect_node 内部处理了。',
            'LangChain 的 AgentExecutor 封装了完整的 ReAct 循环。如果将来你要做更复杂的 Agent，可以用 AgentExecutor + 多个 Tool(向量检索、Web 搜索、数据库查询、代码执行)来替代自己写 LangGraph 节点。但自己写的优势是精确控制每一步--你目前在 agentic_router_graph.py 中的做法更适合学习和理解底层机制。',
        ]),
        ('第五章: Prompt 的评估与迭代', [
            'Prompt Engineering 不是一次写完就完事的。和代码一样，Prompt 需要测试、评估、迭代。一个好的 Prompt 开发流程是: 写初版 -> 跑测试集 -> 分析失败案例 -> 修改 Prompt -> 重新评估。',
            '在你的 evaluate.py 中，三个评估 Prompt(Context Recall、Faithfulness、Answer Relevancy)本身就是 Prompt Engineering 的产物。它们的迭代过程: 先写一版打分标准 -> 跑几条测试 -> 发现 LLM 输出了解释文字而不是纯数字 -> 加「只输出数字、不要任何解释」-> 再跑 -> 发现分数偏差大 -> 细化打分标准。',
            '评估 Prompt 时的一个陷阱: Prompt 写得太好可能导致模型自欺欺人。例如 Faithfulness Prompt 如果描述得太详细，LLM 可能学会了你的评估逻辑，而不是真正的事实核查逻辑。验证方法是: 人工抽查结果，比对 LLM 评分和人工评分的差异。',
        ]),
    ]
)

# ================================================================
# PDF 7: AI Agent 架构
# ================================================================
make_pdf(
    'ai_agent_frameworks.pdf',
    'AI Agent 架构与框架: 从原理到实践',
    'Agent 研究组',
    'AI Agent',
    '高级',
    '2026-07-28',
    [
        ('第一章: 什么是 AI Agent', [
            'AI Agent(人工智能代理)是一个能感知环境、做出决策并执行行动的自主系统。在大语言模型时代，Agent 通常指以 LLM 为大脑、以 Tool 为手脚的智能体系统。',
            '一个典型的 AI Agent 由四部分组成: 大脑(LLM，负责理解和推理)、感知器(从环境获取信息的能力，如搜索引擎、数据库查询、API 调用)、执行器(对环境影响的能力，如发送邮件、执行代码、写入文件)、和记忆(短期记忆如对话上下文、长期记忆如外部知识库)。',
            'AI Agent 和大模型的核心区别: LLM 是回答问题--被动响应；Agent 是完成目标--主动规划。LLM 告诉你应该怎么做，Agent 自己去做。',
        ]),
        ('第二章: 主流 Agent 框架对比', [
            'LangGraph(你正在使用的框架)是 LangChain 团队推出的 Agent 编排框架。它的核心理念是图驱动的 Agent--用有向图定义 Agent 的决策流程，每条边代表一个状态迁移，节点代表一个执行步骤。优势是流程可控、可调试、可预测。',
            'AutoGPT 是最早引发广泛关注的自主 Agent 框架。它的核心是递归循环: 设定目标 -> LLM 拆解任务 -> 调用工具执行 -> 观察结果 -> 评估进展 -> 继续或调整 -> 直到目标达成。AutoGPT 的问题在于自主性过高--容易陷入循环或偏离目标。',
            'CrewAI 专注于多 Agent 协作场景。它模拟团队的运作方式: 每个 Agent 有独立的角色(Role)、目标(Goal)和背景故事(Backstory)，多个 Agent 通过 Task 分配和 Delegate 机制协同工作。适合需要多个专业化 Agent 并行工作的复杂工作流。',
            '选择建议: 学习阶段用 LangGraph(和你的路线完全一致)，简单任务用 LangChain AgentExecutor，复杂项目用 LangGraph + 自定义节点，多 Agent 协作考虑 CrewAI。',
        ]),
        ('第三章: Tool Calling 的深入理解', [
            'Tool Calling 是 Agent 的手。没有 Tool Calling，Agent 只是一个会说话的聊天机器人。有了 Tool Calling，Agent 能搜索最新信息、执行代码、操作文件、调用 API--它的能力边界从「它知道什么」扩展到「它能做什么」。',
            '工具的设计原则: 单一职责(每个工具只做一件事，便于 LLM 准确选择)、清晰的描述(告诉 LLM 这个工具干什么用、什么时候该用它、需要什么参数)、有意义的返回值(返回 LLM 能理解的结果，而不是原始的错误堆栈或状态码)。',
            '在你的项目中，web_retriever.py 的 web_search 和 parent_retriever.py 的 parent_hybrid_retrieve 本质上就是两个 Tool。如果将它们注册为 LangChain Tool 对象，就可以让 Agent 在运行时自主决定用哪个、用几次。',
        ]),
        ('第四章: Agent 的规划能力', [
            '规划(Planning)是 Agent 区别于简单调用-返回模式的核心能力。一个具有规划能力的 Agent 能在执行前先做任务分解、在执行中动态调整计划、在执行后评估结果与目标的差距。',
            'Plan-and-Execute 是最简单的规划模式: 先让 LLM 生成一个完整的执行计划(步骤 1 -> 步骤 2 -> 步骤 3)，然后按部就班执行。优点是结构清晰、容易追踪；缺点是无法应对执行中出现的意外情况。',
            'ReWOO(Reasoning WithOut Observation)是一种更高效的规划模式: Agent 将推理和工具调用分开--推理阶段生成完整计划(不含工具执行)，执行阶段批量调用工具(可并行)，最后基于所有观察结果生成答案。相比 ReAct(每一步都要等待工具返回)，ReWOO 大大减少了 LLM 调用次数。',
        ]),
        ('第五章: Agent 的安全与可靠性', [
            'Agent 的自主性带来了新的安全挑战。一个赋予删除文件权限的 Agent 可能在误判指令后删除关键数据。一个有权调用付费 API 的 Agent 可能在循环中耗尽预付费余额。这些不是理论上的风险，而是实际部署中发生的真实问题。',
            '防御策略包括: 权限最小化(只给 Agent 真正需要的工具)、人工审核(对高风险操作如删除/付费，要求人工确认)、速率限制(限制单位时间内的工具调用次数)、和沙箱隔离(在隔离环境中执行可能有风险的代码)。',
            '你的 agentic_router_graph.py 中的 max_search_attempts=2 就是一种安全控制--通过限制检索轮数上限来防止 Agent 的无限循环。在生产环境中，类似的上限应该设置在所有可能产生循环的地方。',
        ]),
    ]
)

# ================================================================
# PDF 8: 大语言模型基础
# ================================================================
make_pdf(
    'llm_basics.pdf',
    '大语言模型基础知识: 训练、推理与部署',
    'AI 基础学院',
    '大语言模型',
    '初级',
    '2026-03-08',
    [
        ('第一章: 什么是大语言模型', [
            '大语言模型(Large Language Model，LLM)是指参数量巨大(通常在数十亿到数千亿之间)、在大规模文本语料上预训练的深度学习模型。GPT-4、DeepSeek-V4、Claude Sonnet、Qwen 3 等都是主流的大语言模型。',
            'LLM 的核心能力来自于三个因素: 海量的训练数据(通常包括网页、书籍、论文、代码等多来源文本)、巨大的参数规模(更多的参数意味着更强的记忆和泛化能力)、和巧妙的训练方法(从简单的语言建模到复杂的指令微调和人类反馈强化学习)。',
            'LLM 本质上是一个「下一个词预测器」--给定上文，预测最可能的下一个词。这个看似简单的目标，在大规模数据和参数的加持下，涌现出了翻译、推理、代码生成、创意写作等复杂能力。这种现象被称为涌现能力(Emergent Abilities)--模型在达到一定规模后，突然展现出训练目标之外的新能力。',
        ]),
        ('第二章: LLM 的训练流程', [
            '现代 LLM 的训练通常经历三个阶段。第一阶段: 预训练(Pre-training)。在海量文本上训练基础模型，目标是学会语言的结构和知识。这个阶段投入最大--GPT-4 的训练成本估计在 6300 万美元到 1 亿美元之间。',
            '第二阶段: 监督微调(Supervised Fine-Tuning，SFT)。在高质量的人工标注问答数据上微调模型，教会它按指令回答的行为模式。SFT 数据通常包含几十万到几百万条高质量的问答对，每条都需要人工编写或筛选。',
            '第三阶段: 人类反馈强化学习(RLHF)。收集人类对模型输出的偏好数据(对于同一个问题，标注员选择 A 回答还是 B 回答更好)，训练一个奖励模型，然后用强化学习(PPO 算法)优化语言模型使其输出更符合人类偏好。这是让 LLM 从能回答到回答得好的关键一步。',
        ]),
        ('第三章: LLM 的推理与生成', [
            'LLM 的生成过程是典型的自回归(Autoregressive)过程: 逐个 token 预测，每个 token 的预测依赖于之前已生成的所有 token。这解释了为什么 CoT 有效--前面生成的「让我想想...」会影响后面 token 的概率分布。',
            '推理时的关键参数包括: Temperature(控制随机性--低温度让模型更保守、输出更确定；高温度让输出更多样但可能更不稳定)、Top-p(核采样--只从累计概率达到 p 的最可能的 token 中采样)、和 Max Tokens(限制生成长度上限)。',
            '在你的项目中，llm.py 使用了 temperature=0.3--这是一个偏低的值，适合需要事实准确性的 RAG 场景。如果你的场景是创意写作，可以使用 0.7-0.9。',
        ]),
        ('第四章: Token、上下文窗口与成本', [
            'Token 是 LLM 处理文本的最小单位。一个 token 大约对应 0.75 个英文单词或 1-2 个中文字符。「检索增强生成」这 6 个中文字会被 tokenizer 分解为约 4-5 个 token。',
            '上下文窗口(Context Window)是 LLM 在一次推理中能看到的「最大 token 数」。GPT-4 Turbo 的上下文窗口是 128K(约 9.6 万个中文字)，DeepSeek-V4 也是 128K。对于 RAG 系统来说，即使窗口很大，也不建议塞入过多文档--因为 LLM 对长上下文中部的信息关注度会降低(Lost in the Middle 现象)。',
            'API 调用成本由输入 token 和输出 token 分别计费。以 DeepSeek 为例，输入 100 万 token 约 1 元人民币，输出约 2 元。在你的评估脚本中，一道题跑完全流程(6 次 LLM 调用)，假设每次 500 输入 token + 200 输出 token，成本大约在几分钱。',
        ]),
        ('第五章: 开源 vs 闭源模型选型', [
            '闭源模型(如 GPT-4、Claude Sonnet)的优势在于: 开箱即用的高质量、持续的性能提升、稳定的 API 服务。劣势是: 高成本、数据隐私风险、厂商锁定。',
            '开源模型(如 LLaMA、Qwen、DeepSeek)的优势在于: 可控性强、可本地部署保证数据隐私、可微调适配特定领域。劣势是: 需要自建基础设施、模型质量通常不如同期的闭源旗舰。',
            '对于 RAG 学习项目，推荐使用闭源 API(如 DeepSeek，性价比高)快速搭建和验证--你在项目中做的选择完全正确。当管线成熟、需要私有化部署或大规模调用时，再切换到开源模型。',
        ]),
    ]
)

print(f'\nDone! 8 PDFs generated in {OUTPUT_DIR}')
