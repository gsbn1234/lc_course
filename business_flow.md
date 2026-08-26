# Adaptive Research Agent — 完整业务流程与技术解析

> 本文档覆盖当前 Web 版系统的端到端流程：从用户上传 PDF 到最终看到答案，
> 每一步发生了什么、用到了什么技术、背后是什么原理。

---

## 系统全景

```
┌─────────────────┐      HTTP       ┌─────────────────┐     Python 调用    ┌──────────────────┐
│  Streamlit 前端  │ ──────────────→ │   FastAPI 后端   │ ────────────────→ │  LangGraph Agent │
│  (端口 8501)     │ ←────────────── │   (端口 8000)    │ ←──────────────── │  RAG 检索核心     │
└─────────────────┘   SSE 流式返回   └─────────────────┘      graph.stream()  └──────────────────┘
```

三个层次各司其职：

| 层次 | 文件 | 职责 | 用到的技术 |
|------|------|------|-----------|
| 前端 | `app.py` | 展示 UI、上传文件、渲染对话 | Streamlit、requests、SSE 解析 |
| 后端 | `backend.py` | 收请求、管会话、跑 Agent | FastAPI、Pydantic、CORS |
| 核心 | `multi_agent_graph.py` 等 | 检索、搜索、生成答案 | LangGraph、RAG、LLM |

---

## 阶段一：用户上传 PDF（构建知识库）

### 第 1 步：前端收集文件

用户在 Streamlit 页面拖拽 PDF 到 `st.file_uploader`（`app.py` 第 29 行）。

**技术点：`st.file_uploader`** — Streamlit 把用户上传的文件包装成 `UploadedFile` 对象，存在内存里，此时文件还没离开浏览器所在机器。

### 第 2 步：用户点击「构建索引」

`st.button` 返回 `True`，进入 `if` 分支。

**技术点：`st.session_state`** — Streamlit 每次交互重跑整份脚本，但索引等重资源必须跨交互保存，所以存在 `st.session_state` 里。

### 第 3 步：前端组装 multipart 请求（`app.py` 第 52-62 行）

```python
files = [("files", (f.name, f.getvalue(), "application/pdf"))]
requests.post(f"{BACKEND}/api/upload-pdf", files=files)
```

**技术点：multipart/form-data** — HTTP 上传文件的标准格式。每个文件是一个三元组：
- `"files"`：表单字段名，跟后端 `File(...)` 参数名对应
- `f.name`：文件名
- `f.getvalue()`：文件的二进制字节
- `"application/pdf"`：MIME 类型，告诉服务器这是 PDF

### 第 4 步：后端接收并保存（`backend.py` 第 57-75 行）

```python
async def upload_pdf(files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())[:8]       # 生成会话 ID
    content = await f.read()                  # 读二进制
    with open(file_path, "wb") as f_out:      # 写临时文件
        f_out.write(content)
```

**技术点：**
- **FastAPI 的 `UploadFile`** — 异步读取上传的文件，比 `File` 更适合大文件
- **`uuid.uuid4()`** — 生成全局唯一 ID，每个用户一个独立会话
- **`"wb"` 二进制写** — PDF 是二进制文件，必须用二进制模式，不能用文本模式（会破坏文件结构）

### 第 5 步：PDF → Document 列表（`backend.py` 第 77-82 行）

```python
loader = PyPDFLoader(file_path)
docs.extend(loader.load())
```

**技术点：PyPDFLoader** — 把 PDF 每一页解析成一个 LangChain `Document` 对象，带 `page_content`（页文本）和 `metadata`（来源、页码）。

### 第 6 步：父子分块（`backend.py` 第 89-91 行）

```python
child_docs, parent_docs = split_parent_child(docs, child_size=200, parent_size=800, overlap=50)
```

**技术点：Parent Document 检索（父子分块）** — 同一份文档切两次：
- **child 块（200 字符）**：用于检索。小块语义聚焦，向量相似度计算更精准
- **parent 块（800 字符）**：用于喂给 LLM。大块上下文完整，生成质量高

这是 RAG 领域解决"检索精度 vs 上下文完整性"矛盾的经典方案。小块负责找得准，大块负责讲得全。

### 第 7 步：建两个索引（`backend.py` 第 94-97 行）

```python
embeddings = get_embeddings()                          # BGE 嵌入模型
vector_store = get_vector_store(child_docs, embeddings) # FAISS 向量库
bm25 = create_bm25(child_docs)                         # BM25 关键词索引
```

**技术点：混合检索（Hybrid Retrieval）** — 两个索引互补：

| 索引 | 原理 | 擅长 | 短板 |
|------|------|------|------|
| FAISS | 把文本编码成 384 维向量，算余弦相似度 | 语义匹配（"车"能匹配"汽车"） | 专有名词、精确编号 |
| BM25 | jieba 分词后算词频统计 | 关键词精确匹配 | 不理解同义词 |

两者加权融合，权重 `vector_weight=0.6, bm25_weight=0.4`，语义为主关键词为辅。

### 第 8 步：编译 Agent 图（`backend.py` 第 100-103 行）

```python
llm = get_llm()
graph = build_multi_agent_graph(llm, vector_store, bm25, child_docs, parent_docs)
```

**技术点：LangGraph 编译** — 把三个 Agent（Supervisor、Researcher、Writer）注册成节点，用边连成状态图，编译成可执行的图对象。这一步会在"阶段三"详细展开。

### 第 9 步：会话存储（`backend.py` 第 106-113 行）

```python
sessions[session_id] = {"vector_store": ..., "bm25": ..., "graph": ..., ...}
```

**技术点：会话管理** — 每个 session_id 对应一套独立的知识库和图。用户 A 上传的 PDF 不会污染用户 B 的检索。当前用内存字典，生产环境换 Redis。

### 第 10 步：返回 session_id

```json
{"session_id": "753545b7", "page_count": 6, "child_chunks": 46, "parent_chunks": 10}
```

前端把这个 session_id 存进 `st.session_state.session_id`，后续对话都要带上它。

---

## 阶段二：用户提问（流式对话）

### 第 1 步：前端发起流式请求（`app.py` 第 112-121 行）

```python
resp = requests.post(
    f"{BACKEND}/api/chat-stream",
    json={"session_id": ..., "question": prompt, "max_tool_rounds": 5},
    stream=True,      # ★ 关键：流式模式
    timeout=120,
)
```

**技术点：HTTP 流式请求** — `stream=True` 让 `requests` 不等待响应完整返回，而是边接收边处理。这是实现"实时看到搜索过程"的基础。

### 第 2 步：后端解析请求（`backend.py` 第 125-140 行）

```python
async def chat_stream(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        return {"error": "会话不存在"}
    graph = session["graph"]
```

**技术点：Pydantic 模型** — `ChatRequest` 类（第 49-51 行）自动校验请求体。如果前端少传了 `session_id` 或类型不对，FastAPI 自动返回 422 错误，不需要手写校验逻辑。

### 第 3 步：SSE 流式返回（`backend.py` 第 142-171 行）

```python
async def event_stream():
    for chunk in graph.stream(..., stream_mode="values"):
        yield f"data: {json.dumps({...})}\n\n"

return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**技术点：SSE（Server-Sent Events）** — 服务器单向推送技术。每条消息格式固定：

```
data: {"type": "status", "agent": "researcher", "rounds": 1}\n\n
data: {"type": "tool_call", "name": "local_search", "args": {...}}\n\n
data: {"type": "done", "answer": "..."}\n\n
```

- 每条以 `data: ` 开头
- 以两个换行 `\n\n` 结尾
- 前端逐行读，判断 `startswith("data: ")` 后切掉前缀、解析 JSON

为什么用 SSE 而不是一次性返回？因为 Agent 搜索要十几秒，SSE 让用户看到"搜索中→写作中"的动态过程，而不是干等。

### 第 4 步：前端解析事件流（`app.py` 第 124-148 行）

```python
for line in resp.iter_lines():
    line = line.decode()
    if not line.startswith("data: "):
        continue
    data = json.loads(line[6:])   # 切掉 "data: " 前缀
    event_type = data.get("type")
```

**技术点：`resp.iter_lines()`** — 逐行读取流式响应，每行是一个 SSE 事件。三种事件分别处理：
- `status` → 更新界面状态文字（"Researcher 第 N 轮搜索中"）
- `tool_call` → 收集到 `tool_logs` 列表，最后折叠展示
- `done` → 提取最终答案渲染

---

## 阶段三：Agent 内部执行（核心，最值得讲的部分）

这是 `multi_agent_graph.py` 的完整执行流。整个图的结构：

```
[START] → rewrite → supervisor ──派 Researcher──→ researcher ↔ tool（内部循环）
                         ↑                              │ 搜完
                         │←──────────────────────────────┘
                         ↓ 派 Writer
                       writer → supervisor → END
```

### 节点 1：rewrite_query_node（改写问题）

用户问"什么是RAG？"，这个节点调用 LLM 把问题改写成更完整的搜索方向，比如"RAG（检索增强生成）的定义、核心原理、系统架构..."。

同时给三个 Agent 写"任务书"：
- Researcher 收到 SystemMessage（"你是专职研究员，只管搜索"）+ HumanMessage（"搜索指令：..."）
- Writer 的消息通道先空着

**技术点：查询改写（Query Rewrite）** — 用户口语化的提问往往信息量不足，LLM 先扩展成多个搜索角度，提高召回率。

### 节点 2：supervisor_node（第一次调度）

读 `next_agent` 字段，值是 `"researcher"`，打印"派 Researcher 去搜索"，不改状态。

**技术点：Supervisor 模式** — Supervisor 不参与任何推理，只根据 state 里的 `next_agent` 字段做路由决策。这是 Multi-Agent 架构里最经典的"调度者"角色。

### 节点 3-6：Researcher 内部循环（ReAct）

这是整个系统最核心的部分，Researcher 用 ReAct 模式循环搜索：

```
researcher_agent（LLM 思考）→ 输出 tool_calls → researcher_tool_node（执行工具）→ 回到 researcher_agent
```

**第 3 步：Researcher 思考。** `researcher_llm.invoke(messages)` 调用绑定了工具的 LLM。LLM 读完 SystemMessage（"你是研究员"）和 HumanMessage（"搜索指令"），自己决定调什么工具。输出一个 AIMessage，内容为空，但带 `tool_calls`：

```python
tool_calls = [
    {"name": "local_search", "args": {"query": "RAG 检索增强生成 原理"}},
    {"name": "local_search", "args": {"query": "RAG 向量数据库 嵌入"}},
    {"name": "internet_search", "args": {"query": "RAG 应用场景"}},
]
```

**技术点：bind_tools + 并行工具调用** — `llm.bind_tools(search_tools)` 把工具 schema 注入 LLM 的请求，LLM 看到可用工具列表后自主决定调哪个、调几个。一次可以并行输出多个 tool_calls。

**第 4 步：执行工具。** `researcher_tool_node.invoke(...)` 解析 tool_calls，逐个执行。

`local_search` 内部是一条完整的 RAG 检索管线：

```
local_search(query="RAG 检索增强生成 原理")
    │
    ├─ parent_hybrid_retrieve()     混合检索（FAISS + BM25）
    │     在 child 块上搜，通过 parent_id 映射回 parent 大块
    │
    ├─ rerank()                     CrossEncoder 重排
    │     用 bge-reranker 对检索结果两两打分，取 top-3
    │
    └─ compress_documents()         上下文压缩
          把每个文档拆成句子，逐句打分，保留 top-50% 相关句子
```

**技术点（RAG 检索三件套）：**
- **混合检索**：FAISS 语义 + BM25 关键词，加权融合
- **重排（Rerank）**：粗排（向量检索）只保证"大致相关"，精排（CrossEncoder）保证"最相关的排最前"。CrossEncoder 把 query 和 doc 拼一起过 transformer，比向量余弦相似度精准得多，但慢，所以只对 top-10 用
- **上下文压缩**：检索到的 parent 块可能有冗余，拆句打分后只保留相关句子，省 token 又提高精度

**第 5 步：Researcher 再看结果，决定继续还是收手。** 工具执行完，ToolMessage 追加到消息列表，回到 researcher_agent。LLM 读搜索结果，判断"信息够了没"。够了就输出一条有 content 无 tool_calls 的 AIMessage（研究材料总结），不够就再输出新的 tool_calls 继续搜。

**技术点：`researcher_rounds` 防死循环** — 如果 LLM 一直觉得"不够"，会无限搜下去。代码用 `max_tool_rounds=5` 限制，到达上限后强制 `bind_tools([])`（禁用工具）让 LLM 必须总结收手。

### 节点 7：supervisor_node（第二次调度，打包）

Researcher 收手后 `next_agent="writer"`，回到 Supervisor。Supervisor 从 `researcher_messages` 里提取 Researcher 的最终研究材料，装进 `writer_messages`：

- Writer 收到 SystemMessage（"你是写作者，基于材料写答案，别编造"）
- Writer 收到 HumanMessage（"用户问题 + 研究材料 + 请撰写"）

**技术点：上下文隔离** — Researcher 的消息通道里有大量原始搜索结果和 tool_calls，Writer 只看整理好的材料。各看各的，不互相污染。

### 节点 8：writer_agent_node（写作）

`llm.invoke(writer_messages)` — 注意这里用的是**没绑工具的原始 llm**，Writer 不需要搜索能力，只需要写作能力。输出最终答案。

### 节点 9：supervisor_node（收尾）→ END

`next_agent="finish"`，Supervisor 打印"任务完成"，图结束。

---

## 阶段四：答案返回前端

`extract_answer(result)` 从 `writer_messages` 里找最后一条有 content 的 AIMessage，返回给前端渲染。同时 `tool_logs` 里的搜索轨迹用 `st.expander` 折叠展示。

---

## 技术全景图（面试速记版）

| 技术 | 用在哪 | 解决什么问题 |
|------|--------|-------------|
| **Streamlit** | `app.py` | 快速搭 Web 界面，Python 函数式 UI |
| **session_state** | `app.py` | 跨交互保持状态，避免重复建索引 |
| **requests 流式** | `app.py` | 客户端接收 SSE 流 |
| **FastAPI** | `backend.py` | 把 Agent 封装成 REST API |
| **Pydantic** | `backend.py` | 请求体自动校验 |
| **SSE** | `backend.py` + `app.py` | 服务器实时推送搜索进度 |
| **CORS** | `backend.py` | 跨端口（8501→8000）访问 |
| **会话管理** | `backend.py` | uuid + dict，多用户隔离 |
| **LangGraph** | `multi_agent_graph.py` | 多 Agent 编排、状态图、条件路由 |
| **Supervisor 模式** | 同上 | 调度者与执行者分离 |
| **ReAct 模式** | 同上 | LLM 自主决定调什么工具、调几次 |
| **bind_tools** | 同上 | 让 LLM 知道有哪些工具可用 |
| **父子分块** | `parent_splitter.py` | 检索精度 vs 上下文完整性的平衡 |
| **FAISS** | `vector_db.py` | 向量语义检索 |
| **BM25** | `bm25.py` | 关键词精确检索 |
| **混合检索** | `parent_retriever.py` | 语义 + 关键词互补 |
| **CrossEncoder 重排** | `reranker.py` | 精排，提高 top-k 精度 |
| **上下文压缩** | `context_compressor.py` | 去冗余，省 token |
| **Tavily** | `web_retriever.py` | 互联网实时搜索 |
| **DeepSeek API** | `llm.py` | 底层大模型 |
