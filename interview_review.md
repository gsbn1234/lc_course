# Adaptive Research Agent — 面试复习手册

> 基于你亲手写的 `D:\python2\lc_course` 代码生成。每个阶段的解释、数据流、设计决策都来自你的代码，不是通用概念。

---

## 项目整体架构（面试开场白用）

```
你的项目分三层：

第一层：基础设施（main.py 启动时执行一次）
  PDF加载 → 父子切分 → FAISS向量库 + BM25索引 → DeepSeek LLM

第二层：检索增强链路（router_graph.py / agentic_router_graph.py）
  问题改写 → [HyDE] → 路由判断 → 本地/网页检索 → 重排 → 压缩 → 生成

第三层：评估体系（evaluate.py / eval_compare.py）
  8道测试题 → 三个LLM评委 → Context Recall + Faithfulness + Answer Relevancy
```

**你使用的模型和工具：**
- Embedding: `BAAI/bge-small-zh-v1.5`（HuggingFace 本地加载）
- Reranker: `BAAI/bge-reranker-base`（CrossEncoder 本地加载）
- LLM: `deepseek-v4-flash`（通过 DeepSeek API 调用）
- 向量库: FAISS（本地持久化到 `faiss_db/` 目录）
- 关键词检索: BM25Okapi（jieba 分词）
- 网页搜索: Tavily API

---

## 阶段一：基础 RAG + 混合检索

### 涉及文件
`retriever.py`、`hybrid_rag.py`、`rag.py`、`embedding.py`、`bm25.py`、`vector_db.py`、`loader.py`、`config.py`、`llm.py`、`prompt.py`

### 做了什么
从 PDF 加载文档 → RecursiveCharacterTextSplitter 切分（chunk_size=500, overlap=50）→ BGE Embedding → FAISS 向量库 + BM25 索引 → 向量检索和关键词检索的分数加权融合（vector_weight=0.6, bm25_weight=0.4）→ DeepSeek 生成回答。

### 关键代码路径（retriever.py 第 96-209 行）

```
vector_search(question, k=5)           → [(doc, similarity), ...]
bm25_search(question, k=5)             → [(doc, normalized_score), ...]
        ↓
hybrid_retrieve() 融合：
  - 同文档取 max(向量分, BM25分)
  - 不同文档各自保留
  - 加权公式：向量分×0.6，BM25分×0.4
        ↓
multi_hybrid_retrieve()：
  - 对多个 query 各搜一轮
  - 跨 query 累加分数（同一文档被多 query 命中 = 加分）
```

### 设计决策

**为什么 vector_weight=0.6 而不是 0.5？**
向量检索在语义理解上优于 BM25，中文同义词和近义词场景下向量检索召回率更高，所以给向量稍高的权重。但没有做严格的超参调优——面试官如果追问"怎么验证 0.6 是最优的"，可以承认这是经验值，更好的做法是用评估集的 Context Recall 做网格搜索。

**为什么相似度用 1/(1+score) 而不是余弦相似度？**
FAISS 的 `similarity_search_with_score` 返回的是 L2 距离（越小越相似），不是余弦相似度。1/(1+score) 是一个简单的归一化：距离 0 → 相似度 1.0，距离 ∞ → 相似度趋近 0。面试官如果问"为什么不用 inner product"，可以答：BGE 模型推荐用 L2 距离，保持和模型训练目标一致。

**为什么 BM25 要做归一化（line 77：score/max_score）？**
BM25 原始分数范围不固定（取决于文档长度和词频），不归一化的话 0.6/0.4 的融合权重就没意义了。归一化到 [0, 1] 后两个检索引擎的分数才可比。

---

## 阶段二：Parent Document Retrieval

### 涉及文件
`parent_splitter.py`、`parent_retriever.py`

### 做了什么
用两个不同的 chunk_size 切分同一组 PDF：
- Child（200 字符）：建 FAISS + BM25 索引，用于精确检索
- Parent（800 字符）：完整上下文，用于 LLM 生成

检索时在 child 粒度搜，通过 `parent_id` 映射回 parent 大块。解决了经典矛盾：小块搜得准但上下文不完整，大块上下文完整但搜不准。

### 关键代码路径（parent_splitter.py 第 5-51 行）

```
PDF Documents
    │
    ├── parent_splitter (800字) → parent_docs，每块有唯一 parent_id
    │
    └── 对每个 parent 再用 child_splitter (200字) 切 → child_docs
        child_docs 继承对应 parent 的 parent_id
```

### 关键代码路径（parent_retriever.py 第 7-93 行）

```
parent_hybrid_retrieve():
    在 child 粒度上用 vector_search + bm25_search
        ↓
    按 parent_id 分组（多个 child 可能映射到同一 parent）
        ↓
    parent 去重，取最高 child 分
        ↓
    返回 top_k 个 parent Document（800 字版本）
```

### 设计决策

**为什么 child=200, parent=800？**
BGE-small-zh-v1.5 的最大序列长度是 512 tokens。200 汉字约 400-500 tokens，基本填满模型窗口，Embedding 质量最高。800 汉字是 DeepSeek 上下文窗口完全能容纳的，同时给 LLM 足够的上下文做判断。没有严格的数学推导，是基于模型能力的工程取舍。

**去重逻辑为什么用 page_content 的 hash 而不是 parent_id？**
parent_retriever.py 第 73 行用的是 `hashlib.md5(parent_doc.page_content.encode()).hexdigest()`。因为多个不同的 parent 可能有相同的 page_content（splitter 在边界情况可能产生重复），用 parent_id 去重不如用内容去重可靠。

---

## 阶段三：Query Routing（路由判断）

### 涉及文件
`router_graph.py`

### 做了什么
在检索前加一个路由判断：LLM 看用户问题 → 输出 "local"（查本地 PDF）或 "web"（调 Tavily 搜互联网）。两条路径用相同的后续节点（重排 → 压缩 → 生成）。

### 图结构（router_graph.py）

```
[START]
    │
    ▼
rewrite_query_node       ← 问题改写
    │
    ├── [use_hyde?] → hyde_node → 假想答案
    │
    ▼
classify_route_node      ← LLM 判断 local/web
    │
    ├─ local ──→ local_retrieve_node
    └─ web ────→ web_retrieve_node
    │                 │
    └────────┬────────┘
             ▼
    rerank_compress_node
             │
             ▼
       generate_node
```

### 关键代码路径（router_graph.py 第 77-86 行）

```python
def classify_route(question, llm):
    chain = router_prompt | llm
    response = chain.invoke({"question": question})
    route = response.content.strip().lower()
    if route not in ("local", "web"):
        route = "local"  # 兜底
    return route
```

### 设计决策

**为什么用一个专门的 LLM 调用来做路由，而不是用关键词规则？**
关键词规则（"最新"→web）覆盖不了"2026年7月人工智能领域有什么重大新闻"这种没有明确触发词但需要实时信息的问题。LLM 做路由虽然多一次调用（约 0.1 秒），但准确率高很多。

**为什么 local 检索用 Multi-Query + Parent Hybrid Retrieve，而 web 只用单一 Tavily 搜索？**
router_graph.py 第 131-152 行，local_retrieve_node 调了 `generate_queries` 生成 3 个多角度查询，每个都搜一轮。web_retrieve_node 只用 `rewrite_question` 搜一次 Tavily。原因是：本地库是静态文档，多角度查询能提高召回；Tavily 已经内部做了 query 优化，重复搜 3 次不如直接信任搜索引擎的结果。

**为什么路由失败时默认走 local 而不是 web？**
因为你的知识库是主力，web 是补充。路由失败时走 local 的代价（搜不出结果）远小于走 web 的代价（返回无关信息 + 浪费 Tavily API 额度）。

---

## 阶段四：HyDE（假想文档检索）

### 涉及文件
`hyde.py`

### 做了什么
LLM 根据用户问题编一段 150-250 字的"假答案"（技术白皮书风格），用这个假答案的文本作为查询词去向量检索。利用的是：假答案的文体（陈述句）= 真实文档的文体，所以向量更接近文档空间。

### 关键代码路径（hyde.py 第 19-50 行）

```
用户问题: "什么是RAG?"
    ↓ hyde_prompt
LLM 生成: "RAG（检索增强生成）是一种将信息检索与语言模型生成阶段相融合的架构范式..."
    ↓ Embedding(假答案)
假答案向量 → FAISS 搜索 → 文档空间中最近的真实文档
```

### 设计决策

**为什么用 150-250 字而不是更长？**
太长会增加 LLM 调用延迟（约 0.5 秒），太短信息密度不够。150-250 字足够包含 3-5 个关键词，能让 Embedding 落在正确的语义区域。

**为什么 prompt 里要求"像技术白皮书那样写"？**
因为你的 PDF 就是技术文档（向量数据库、RAG、Trans…er 等），让假答案模仿技术文档的文体，能最大化向量空间对齐效果。如果 PDF 是小说，就要让假答案模仿小说文体。

**HyDE 的边界在哪儿（什么时候反而降效果）？**
你用 eval_compare.py 跑过对比实验。简单的事实性问题（"RAG 能解决什么问题"），HyDE 的假答案可能编造一些训练数据里的信息，把检索带偏到知识库里不存在的内容上。Answer Relevancy 在简单问题上可能下降。所以 HyDE 更适合开放性的、需要深度解释的问题，不适合精确事实检索。

---

## 阶段五：Multi-Query（多角度查询）

### 涉及文件
`multi_query.py`

### 做了什么
LLM 根据用户问题生成 3 个不同角度的查询问题，各自独立检索后合并去重，同一文档被多个查询命中则累积分数。

### 关键代码路径（multi_query.py 第 33-93 行）

```
rewrite_question: "RAG技术原理"
    ↓ multi_query_prompt
LLM 生成 3 个查询:
  1. "RAG 检索增强生成 工作流程 索引 检索 生成"
  2. "RAG 如何缓解大模型幻觉问题"
  3. "Retrieval Augmented Generation 架构详解"
    ↓ 各自 parent_hybrid_retrieve
合并去重（累加分数）→ 分数高的文档被多个查询共同命中
```

### 设计决策

**为什么是 3 个查询不是 5 个？**
每多一个查询 = 多一轮 parent_hybrid_retrieve = 多 2×(5+5) 次检索操作 + 额外的 LLM token 消耗。3 个在"覆盖面提升"和"延迟增加"之间是较好的折中。5 个会有边际效应递减（第 4、5 个查询往往和前面重复度高）。

**查询合并时为什么累加分数而不是取最大值？**
取 max 的话，一个文档被 3 个查询命中一次和被 1 个查询命中一次分数一样。累加能反映出"多个角度都认为这个文档相关"的信号，排序质量更高。

**为什么生成的查询里可能有英文关键词（如 "Retrieval Augmented Generation"）？**
因为 BGE Embedding 模型在训练时见过中英混合语料，英文技术术语的 Embedding 质量不输中文。不加语言限制反而能利用模型的多语言能力。

---

## 阶段六：Agentic RAG（反思循环）

### 涉及文件
`agentic_router_graph.py`

### 做了什么
在 rerank_compress_node 后面加了一个 reflect_node，LLM 审视检索到的上下文 → 判断"够不够"→ 不够就生成新的搜索词 → 回到 local_retrieve_node 再搜 → 循环。和 router_graph.py 最大的区别：router_graph 是"搜一次 → 回答一次"，agentic_router_graph 是"搜 → 反思 → 搜 → 反思 → 够了一 停下来回答"。

### 图结构（agentic_router_graph.py）

```
[START]
    │
    ▼
rewrite_query_node
    │
    ├── [use_hyde?] → hyde_node
    │
    ▼
classify_route_node
    │
    ├─ local ──→ local_retrieve_node  ←──────────┐
    └─ web ────→ web_retrieve_node                │
    │                 │                           │
    └────────┬────────┘                           │
             ▼                                    │
    rerank_compress_node                          │
             │                                    │
             ▼                                    │
       reflect_node  ← 审视上下文                  │
             │                                    │
        ┌────┴────┐                               │
        │ done     │ search: <新搜索词> ────────────┘
        ▼
  generate_node → [END]
```

### 核心新增字段（agentic_router_graph.py 第 59-73 行）

```python
class AgenticRAGState(TypedDict):
    # ... 原有字段 ...
    search_attempts: int        # 已搜几轮（防无限循环）
    reflection_query: str       # reflect 生成的新搜索词
    accumulated_docs: List[dict] # 多轮累积池（所有轮的文档一起重排）
```

### 关键代码路径

**reflect_node（第 283-324 行）核心逻辑：**

```
contexts_text = 拼上 final_docs 的 page_content（截断到 500 字符）
    ↓ reflect_prompt
LLM 输出: "done" 或 "search: <新搜索词>"
    ↓
"search:" → 设置 route="retrieve_again" + reflection_query=新搜索词
"done"   → 设置 route="generate"
```

**local_retrieve_node 的双模检索（第 149-247 行）：**

```
模式 A（首轮）：HyDE/Multi-Query 多角度检索
模式 B（Agent 再检索）：直接用 reflection_query 单一查询检索
    因为 Agent 生成的新搜索词已经很精准，不需要再发散
```

**rerank_compress_node 的关键变化（第 259-280 行）：**
用 `accumulated_docs`（所有轮）而不是 `docs`（当前轮）做重排。所以搜了两轮的话，两轮的文档一起参与 CrossEncoder 打分。

### 设计决策

**为什么 max_search_attempts=2 而不是 5？**
每次搜索都走一遍 parent_hybrid_retrieve + rerank + compress + LLM reflect，延迟约 2-3 秒。2 轮 = 总延迟 4-6 秒，用户可以接受。5 轮 = 10-15 秒，体验差。另外 2 轮已经覆盖了"第一轮概览 + 第二轮针对性补充"的模式，再多轮的边际收益很低。

**为什么 reflect 后再搜只能走 local_retrieve_node，不能切 web？**
agentic_router_graph.py 第 356-358 行：
```python
if state.get("route") == "retrieve_again":
    return "local_retrieve_node"
```
这是一个工程简化。reflect 时 original route 已经消费掉了，没有保存"第一轮走的是 local 还是 web"。如果要支持"再搜时也能切 web"，需要在 State 里加一个 `original_route` 字段。这是面试时可以主动提出的改进点。

**为什么要用 accumulated_docs 而不是只保留当前轮的 docs？**
单个检索词只能命中一个角度。第一轮检索词是"RAG 架构"，第二轮 Agent 说"不够，搜'RAG 实际应用案例'"。如果只保留第二轮的结果，第一轮搜到的 RAG 架构内容就丢了。accumulated_docs 把两轮结果合并，reranker 从中挑最好的，不会丢失信息。

---

## 阶段七：Self Query Retriever

### 涉及文件
`self_query.py`

### 做了什么
在检索前加一层 LLM 查询分析：从用户自然语言问题中提取 `semantic_query`（纯语义查询词）+ `filter_dict`（结构化过滤条件），然后用 semantic_query 检索多份结果（top_k×3），再用 filter_dict 做 metadata 过滤，最后截断到 top_k。

### 关键代码路径（self_query.py 第 92-131 行）

```
用户: "找向量数据库分类下的中级文档"
    ↓ extract_query_and_filter()
LLM 输出:
  semantic_query: "向量数据库原理和应用"
  filter_dict: {"subject": "向量数据库"}
    ↓
parent_hybrid_retrieve(semantic_query, top_k=30)  ← 搜 3 倍
    ↓
apply_metadata_filter(docs, filter_dict)  ← 从 30 条筛出 subject 含"向量数据库"的
    ↓
返回前 10 条
```

### 可过滤的元数据字段（self_query.py 第 27-43 行）

| 字段 | 含义 | 示例值 |
|------|------|--------|
| source | PDF 文件名 | `vector_databases.pdf` |
| subject | 主题分类 | 向量数据库、RAG、大语言模型 |
| author | 作者/研究组 | 向量数据库研究组 |

### 设计决策

**为什么先搜多再过滤，而不是在 FAISS 检索时就用 filter？**
你的 metadata（subject、author）存在 Document 的 metadata 属性里，但 FAISS 索引只存储向量，不存储 metadata。FAISS 的 `similarity_search` 不支持在检索时做结构化过滤。所以只能"搜完再筛"——搜 3 倍数量 = 给过滤留足够的候选池。

**如果 filter_dict 是空的（用户没指定过滤条件），self_query 还有意义吗？**
self_query.py 第 209 行：
```python
fetch_k = top_k * 3 if filter_dict else top_k
```
没有 filter 时，self_query 退化为普通检索——只做了 semantic_query 提取，没做过滤。多了一次 LLM 调用（extract_query_and_filter），但语义查询词比原始问题更聚焦。如果面试官问"这样是不是浪费了 LLM 调用"，可以答：semantic_query 抽取本身也有价值——它去掉了用户问题里的口语化表达和冗余信息。

**为什么过滤用的是子串匹配（`value.lower() not in meta_value.lower()`）而不是严格相等？**
因为 PDF 的 metadata 值可能有前后空格、拼写变体。比如 subject 可能是"向量数据库"也可能是"向量数据库技术"。子串匹配更鲁棒。代价是可能误匹配（"大语言模型"匹配到"大语言模型应用"），但在当前 8 个 PDF 的规模下误匹配概率很低。

---

## 跨阶段知识：评估体系

### 涉及文件
`evaluate.py`、`eval_compare.py`、`eval_dataset.py`

### 三个评估指标

| 指标 | 评的是什么 | 谁和谁比 |
|------|-----------|---------|
| Context Recall | 检索够不够全 | context vs ground_truth |
| Faithfulness | 回答有没有编造 | answer vs context |
| Answer Relevancy | 回答有没有跑题 | answer vs question |

这三个指标覆盖了 RAG 管线的三个故障点：检索层（recall）、生成层（faithfulness）、整体（relevancy）。

### 关键设计

**为什么用 LLM-as-Judge 而不是 ragas 等现成库？**
evaluate.py 没引入 ragas。好处是：不依赖第三方库的版本兼容性，prompt 自己写、逻辑自己掌控。代价是：LLM 打分的稳定性不如 ragas 的统计指标（LLM 对同一输入可能两次打出不同分数）。面试官如果问"你评估体系的可靠性怎么保证"，可以承认这是简化方案，更严谨的做法是多次打分取平均 + 人工抽检。

**为什么 eval_compare.py 要构建两个独立的 graph？**
eval_compare.py 第 124-125 行：
```python
graph_baseline = build_rag_graph(..., use_hyde=False)
graph_hyde    = build_rag_graph(..., use_hyde=True)
```
因为 `use_hyde` 会改变图结构（是否注册 hyde_node、边连接方式）。不能在一个 graph 上动态切换，只能构建两个实例，确保对比时每个实例的图结构稳定。

**为什么评估用 graph.invoke 而不是 ask() 函数？**
因为 ask() 是早期 rag.py 里的函数，它用的是旧架构（multi_hybrid_retrieve 而非 parent_hybrid_retrieve）。graph.invoke 走的是 LangGraph 工作流，和 main.py 保持一致，评估结果才能反映线上真实表现。

---

## 辅助模块速查

| 文件 | 作用 | 面试相关度 |
|------|------|-----------|
| `config.py` | 管理 PDF_PATH、API Key、FAISS_PATH | 低 |
| `llm.py` | `ChatOpenAI(model="deepseek-v4-flash", temperature=0.3)` | 中（为什么 temperature=0.3？检索场景需要确定性，不能太发散） |
| `embedding.py` | `HuggingFaceEmbeddings("BAAI/bge-small-zh-v1.5")` | 中（为什么用 bge-small 不用 bge-large？small 更快，中文效果已足够好） |
| `loader.py` | `PyPDFDirectoryLoader` 加载 PDF 目录 | 低 |
| `web_retriever.py` | Tavily 搜索 + Document 包装 | 中（为什么 web 结果 score 统一给 1.0？没有向量分数，Tavily 内部已排序） |
| `reranker.py` | CrossEncoder 重排序 | 高 |
| `context_compressor.py` | 逐句打分、保留 top-50% 句子 | 高 |
| `prompt.py` | 最终回答的 prompt 模板 | 中（为什么要求"仅根据参考资料"？防幻觉，是 RAG 的底线约束） |

---

## 面试高频追问速答

**Q: 你的系统从用户提问到输出答案，延迟大概多少？**
A: 本地检索路径约 3-5 秒（Embedding + FAISS/BM25 + LLM改写 + Multi-Query + Rerank + LLM生成）。Agentic RAG 两轮检索约 6-8 秒。主要瓶颈是 LLM 调用（改写、HyDE、Multi-Query、Reflect、生成共 5-6 次 LLM 调用），而不是检索本身。

**Q: 如果知识库从 8 个 PDF 扩展到 1000 个，哪些地方会出问题？**
A: FAISS 在百万级向量仍然够快（IVF 索引），但 BM25 的线性扫描会变慢（可以考虑 Elasticsearch 替代）。更大的问题是 Multi-Query 召回的东西变多，reranker 的 CrossEncoder 是 O(n) 的 pair 计算，top-10 重排变成 top-100 重排会显著变慢。

**Q: 你的系统怎么处理用户连续追问？**
A: 目前每个问题独立处理，没有记忆。AgenticRAGState 没有存历史对话。如果要支持追问，需要加 Checkpointer（SqliteSaver）+ thread_id 区分会话——这就是接下来要学的 Memory 阶段。

**Q: 如果让你重新设计，有什么会做不一样的？**
A: 一个明显的改进：现在 reflect_node 再搜索只能回 local，实际上如果第一次走的是 web 路由，再搜索也应该能切 web。解决方案是在 State 里保留 initial_route 字段。另外 Self Query 的过滤可以前置到 FAISS 检索阶段（用 FAISS 的 filter 参数，需要额外维护一个 id→metadata 的映射表）。
