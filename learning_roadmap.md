# AI 应用开发 — 四周学习路线（实习冲刺版）

> 目标：一两个月内产出完整可演示的 AI Agent 应用，用于实习面试。
> 前置基础：已掌握 LangGraph 多 Agent 编排、RAG 检索管线（FAISS + BM25 + Rerank + 父子分块）。

---

## 整体节奏

| 周次 | 主题 | 核心交付 | 优先级 |
|------|------|---------|--------|
| 第 1 周 | Streamlit 界面包装 | 可交互的 Web 应用 | 🔴 最高 |
| 第 2 周 | FastAPI 前后端分离 | 前端调后端 API 跑通 | 🟡 高 |
| 第 3 周 | Memory 多轮对话 | 不会失忆的 Agent | 🟡 高 |
| 第 4 周 | MCP 外部工具接入 | 工具生态可扩展 | 🟢 中 |

---

## 第 1 周：Streamlit 界面包装

### 1.1 学习路径

**Day 1（2-3 小时）：** Streamlit 官网 30 分钟教程过一遍，只需要掌握这几个 API：

```
st.chat_input()        # 聊天输入框
st.chat_message()      # 聊天气泡
st.sidebar             # 侧边栏
st.file_uploader()     # 文件上传
st.session_state       # 跨交互保持状态
st.spinner()           # 加载动画
st.expander()          # 折叠面板
st.empty()             # 占位容器（流式输出的关键）
```

不要学其他的。不要碰 `st.dataframe`、`st.plotly_chart`、`st.form`，现阶段用不上。

**Day 2-3（4-6 小时）：** 搭建第一版骨架。

### 1.2 核心代码骨架

```
# app.py 主结构
import streamlit as st

# ========== 1. 初始化 session_state ==========
if "graph" not in st.session_state:
    st.session_state.graph = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "parent_docs" not in st.session_state:
    st.session_state.parent_docs = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 2. 侧边栏：PDF 上传 + 参数配置 ==========
with st.sidebar:
    st.header("知识库")
    uploaded_files = st.file_uploader(
        "上传 PDF", type="pdf", accept_multiple_files=True
    )
    if uploaded_files and st.button("构建索引"):
        with st.spinner("正在处理 PDF..."):
            # 保存上传的 PDF → 跑 load_documents → split_parent_child
            # → get_vector_store → create_bm25
            # → 全部存进 st.session_state
            st.session_state.vector_store = ...
            st.session_state.bm25 = ...
            st.session_state.chunks = ...
            st.session_state.parent_docs = ...
            # 重建 graph
            st.session_state.graph = build_multi_agent_graph(...)
        st.success("索引构建完成")

    st.divider()
    max_rounds = st.slider("最大搜索轮数", 1, 10, 5)

# ========== 3. 主区域：聊天界面 ==========
st.title("Adaptive Research Agent")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("输入你的问题"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        # ... 流式调用 graph.stream()，见 1.3
```

### 1.3 流式输出（本周最关键的技术点）

你现在的 `graph.invoke()` 是同步的——整张图跑完才返回，用户盯着一个 `st.spinner` 等十几秒。必须改成 `graph.stream()`：

```
# graph.stream() 返回的是生成器，每个节点执行完就 yield 一次
with st.chat_message("assistant"):
    status_container = st.empty()  # 占位容器，用来实时更新状态

    result = None
    for chunk in st.session_state.graph.stream(
        {"question": prompt, "researcher_rounds": 0, "max_tool_rounds": max_rounds},
        stream_mode="values",  # values 模式：每个节点后返回完整 state
    ):
        result = chunk  # 保留最后一帧

        # 实时显示 Researcher 状态
        next_agent = chunk.get("next_agent", "")
        rounds = chunk.get("researcher_rounds", 0)

        if next_agent == "researcher":
            status_container.info(f"🔍 Researcher 第 {rounds} 轮搜索中...")
        elif next_agent == "writer":
            status_container.info("✍️ Writer 正在撰写答案...")

    # 流式结束，显示最终答案
    answer = extract_answer(result)
    status_container.empty()  # 清掉状态文字
    st.write(answer)

st.session_state.messages.append({"role": "assistant", "content": answer})
```

`stream_mode="values"` 每经过一个节点就 yield 一次完整的 state 快照。你在界面上拿 `chunk["next_agent"]` 实时更新状态文字，用户体验就从"干等"变成了"看到它在干活"。

如果后续想实现打字机效果（Writer 逐字输出），需要 `stream_mode="messages"` + 解析 AIMessageChunk，这个留到 FastAPI 阶段一起做。

### 1.4 工具调用日志展示

用 `st.expander` 把 Researcher 的搜索过程折叠起来：

```
# 在 graph.stream 的循环中，收集工具调用信息
tool_logs = []
for chunk in graph.stream(...):
    messages = chunk.get("researcher_messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls"):
            for tc in last_msg.tool_calls:
                tool_logs.append(f"🔧 {tc['name']}({tc['args']})")

# 流式结束后展示
with st.expander("查看检索过程"):
    for log in tool_logs:
        st.text(log)
```

### 1.5 本周的坑

1. **FAISS 索引每次交互都重建。** 你原来 main.py 的加载逻辑放在脚本顶层，Streamlit 每次点按钮都重跑整份脚本。必须把索引放进 `st.session_state`，只在"构建索引"按钮触发时跑一次。

2. **graph.stream() 第一次上手会报 `graph.stream() got unexpected keyword argument`。** 检查你用的 LangGraph 版本，`stream_mode` 从 0.1.0 开始支持，如果版本太老先 `pip install -U langgraph`。

3. **不要在 chat_message 里面再嵌套 chat_message。** 流式更新时用 `st.empty()` 占位 + `.write()` 更新，不要重复创建 chat_message。

---

## 第 2 周：FastAPI 前后端分离

### 2.1 为什么要拆

你现在 Streamlit 直接调 Python 函数——界面和 Agent 逻辑糊在一起。拆开之后：

- 前端只负责 UI（显示、上传、输入）
- 后端只负责 Agent（检索、搜索、生成）

面试官看到你会做前后端分离 + 流式 SSE，就默认你具备"把 AI 模型部署成服务"的能力——这是 AI 应用开发岗位的基本要求。

### 2.2 接口设计

```
POST  /api/upload-pdf      # 上传 PDF，返回 session_id
POST  /api/chat-stream     # 流式对话（SSE），输入 session_id + question
GET   /api/health          # 健康检查
```

### 2.3 后端 skeleton

```
# backend.py
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

# 每个用户独立的知识库实例
sessions = {}  # {session_id: {vector_store, bm25, chunks, parent_docs, graph}}

class ChatRequest(BaseModel):
    session_id: str
    question: str
    max_tool_rounds: int = 5

@app.post("/api/upload-pdf")
async def upload_pdf(files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())[:8]

    # 1. 保存上传的 PDF 到临时目录
    # 2. load_documents → split_parent_child
    # 3. get_vector_store → create_bm25
    # 4. build_multi_agent_graph

    sessions[session_id] = {
        "vector_store": vector_store,
        "bm25": bm25,
        "chunks": child_docs,
        "parent_docs": parent_docs,
        "graph": graph,
    }

    return {"session_id": session_id, "chunk_count": len(child_docs)}

@app.post("/api/chat-stream")
async def chat_stream(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        return {"error": "session not found"}

    graph = session["graph"]

    async def event_stream():
        for chunk in graph.stream(
            {"question": req.question, "researcher_rounds": 0, "max_tool_rounds": req.max_tool_rounds},
            stream_mode="values",
        ):
            # SSE 格式：data: {json}\n\n
            yield f"data: {json.dumps({'status': chunk.get('next_agent', '?'), 'rounds': chunk.get('researcher_rounds', 0)})}\n\n"

        # 提取最终答案
        answer = extract_answer(chunk)
        yield f"data: {json.dumps({'status': 'done', 'answer': answer})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}
```

### 2.4 前端改造

Streamlit 前端不再直接调 `graph.invoke()`，而是调你的 FastAPI：

```
# 上传 PDF
files = {"files": pdf_bytes}
resp = requests.post("http://localhost:8000/api/upload-pdf", files=files)
session_id = resp.json()["session_id"]

# 流式对话
resp = requests.post(
    "http://localhost:8000/api/chat-stream",
    json={"session_id": session_id, "question": prompt},
    stream=True,
)
for line in resp.iter_lines():
    if line:
        data = json.loads(line.decode().replace("data: ", ""))
        # 更新界面状态
```

### 2.5 本周的坑

1. **session 字典存在内存里，重启后端就没了。** 现阶段可以接受——面试 demo 用。正式上线要换成 Redis，但这个不在本周范围。

2. **Streamlit 读 SSE 流要处理 `requests.iter_lines()`。** 不要用 `resp.json()`——它等整个响应返回才解析，流式效果就没了。

3. **CORS 跨域。** Streamlit 默认跑在 8501 端口，FastAPI 跑在 8000 端口，需要加 `fastapi.middleware.cors.CORSMiddleware`，否则浏览器拦截。

4. **文件上传的 PDF 不能直接从内存读。** FastAPI 的 `UploadFile` 返回的是字节流，你的 `load_documents`（PyPDFDirectoryLoader）需要物理路径。先把上传的文件 `await file.read()` 然后写到临时目录。

---

## 第 3 周：Memory 多轮对话

### 3.1 两套方案

#### 入门版（本周主力）

在 `MultiAgentState` 里加一个字段，把历史问答塞进 Writer 的输入：

```
# multi_agent_graph.py 的 MultiAgentState 加一行
class MultiAgentState(TypedDict):
    ...
    conversation_history: Annotated[list, operator.add]  # 新增
```

`rewrite_query_node` 改造：把历史问答格式化后注入 `writer_human`：

```
def rewrite_query_node(state, llm):
    ...
    # 从 conversation_history 里取最近 N 轮
    history = state.get("conversation_history", [])
    history_text = ""
    for item in history[-6:]:  # 只取最近 3 轮问答
        history_text += f"用户：{item['question']}\n助手：{item['answer'][:200]}...\n\n"

    writer_human = HumanMessage(
        content=f"对话历史：\n{history_text}\n\n用户问题：{question}\n\n研究材料：...\n\n请撰写回答。"
    )
```

main.py 里每次问答结束后存档：

```
result = graph.invoke({...})
answer = extract_answer(result)
# 存历史
history = result.get("conversation_history", [])
history.append({"question": q, "answer": answer})
```

这样用户问"它和传统搜索有什么区别"时，Writer 能看到上一轮聊的是 RAG，自然就接上了。

缺点：历史消息越长，Prompt 越长，token 开销越大。限制最近 3 轮即可。

#### 进阶版（可选，有空再看）

LangGraph 原生 Checkpointer：

```
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# invoke 时传 thread_id
graph.invoke(
    {"question": "什么是RAG？", ...},
    config={"configurable": {"thread_id": "user-session-123"}},
)
```

Graph 自动记住这个 thread_id 的完整历史，下次 invoke 同一个 thread_id，LLM 能看到之前所有消息。但 **Checkpointer 和你的自定义 State（researcher_messages、writer_messages）怎么配合，需要单独研究**，不是一两天能搞定的。先做入门版，够用了。

### 3.2 本周的坑

1. **conversation_history 不要无限增长。** 限制最近 3-6 轮，否则 Prompt 超长、token 费用爆炸。
2. **历史回答建议截断到 200 字符。** 完整的旧回答塞进去只是浪费上下文，摘要足够 LLM 理解前面聊了什么。
3. **Checkpointer 用的是 LangGraph 自带的消息通道（`messages`），不是你自己定义的 `researcher_messages`。** 如果要用 Checkpointer，你的多通道设计需要改，成本很高。这也是我让你先做入门版的原因。

---

## 第 4 周：MCP 外部工具接入

### 4.1 核心思路

你现在的 Researcher 有两个工具：`local_search`、`internet_search`。MCP 让你能接入别人写的工具——比如文件读取、数据库查询、Slack 消息——而不需要自己写每个工具的代码。

核心链路：

```
MCP Server（别人提供的标准工具）
    │  JSON-RPC 协议
    ▼
MCP Client（获取工具列表 + 描述 + 参数 schema）
    │
    ▼
适配层（MCP Tool → LangChain BaseTool）
    │
    ▼
search_tools = [local_search, internet_search, mcp_file_reader, mcp_folder_lister]
    │
    ▼
researcher_llm.bind_tools(search_tools)  ← 你现有代码，一行不改
```

### 4.2 实操步骤

**Step 1：** 启动一个本地 MCP 文件系统 Server。anthropic 官方有现成的 `@anthropic/mcp-server-filesystem`，或者用 Python 社区的替代。它暴露两个工具：`read_file` 和 `list_directory`。

**Step 2：** 在 Python 里用 `mcp` SDK 连接这个 Server：

```
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="npx", args=["-y", "@anthropic/mcp-server-filesystem", "/path/to/docs"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        tools = await session.list_tools()  # 拿到工具列表
```

**Step 3：** 写适配器，把 MCP 工具转成 LangChain `@tool`：

```
from langchain_core.tools import tool

def mcp_tool_to_langchain(mcp_tool, session):
    """把一个 MCP 工具包装成 LangChain BaseTool"""

    @tool
    def wrapper(**kwargs):
        """执行 MCP 工具调用"""
        import asyncio
        result = asyncio.run(session.call_tool(mcp_tool.name, kwargs))
        return result.content[0].text

    wrapper.name = mcp_tool.name
    wrapper.description = mcp_tool.description
    # 参数 schema 从 mcp_tool.inputSchema 转过来
    return wrapper
```

**Step 4：** 把新工具注入 search_tools：

```
# build_multi_agent_graph 里加几行
mcp_tools = get_mcp_tools()  # 从 MCP Server 获取并转换
all_tools = [local_search, internet_search] + mcp_tools
researcher_llm = llm.bind_tools(all_tools)
```

### 4.3 本周的坑

1. **MCP 工具调用是 async 的，ToolNode 内部是 sync 的。** 你的适配器里需要 `asyncio.run()` 包一层，或者把 `researcher_tool_node` 改成 async。这是一个实际会卡住的问题，不要低估。

2. **MCP Server 进程管理。** 你启动的 MCP Server 是独立进程，FastAPI 重启或 Streamlit 重跑时，要注意进程的启动和销毁，避免僵尸进程。

3. **工具数量增加后，Researcher 的决策质量可能下降。** LLM 面对 5+ 个工具时，选错工具的概率上升。工具 description 要写得足够清晰，明确"什么场景用这个，什么场景不要用"。

4. **不要试图接入太多 MCP 工具。** 本周目标只是完成 1-2 个 MCP 工具的接入，证明"你的 Agent 架构支持外部工具扩展"。能跑通比做多重要。

---

## 四周之后：持续迭代路线

这些不在紧凑的四周计划内，但知道方向：

| 阶段 | 做什么 | 什么时候做 |
|------|--------|-----------|
| Docker 打包 | FastAPI + 向量库 + MCP Server 一键部署 | 拿到面试邀请后，部署到云上给面试官试 |
| LangGraph 子图 | 把 Researcher 内部的搜索逻辑拆成子图，更模块化 | 项目迭代期 |
| Human-in-the-Loop | 用户可以在 Researcher 搜索结果页面点"继续搜"或"够了" | 有前端经验后 |
| 前端替换 | Streamlit 换 Vue/React，做更专业的 UI | 找到实习后，入职前 |
| Redis 持久化 | session 字典换成 Redis，支持重启不丢、多用户并发 | 入职后 |

---

## 每天怎么安排

工作日每天保证 2 小时有效时间，周末 4-6 小时：

- 前 20 分钟：看文档、理解概念
- 中间 80 分钟：动手写代码，写一点跑一点，不攒到最后
- 最后 20 分钟：记录踩过的坑，更新到笔记里

遇到具体报错直接来问我，不要卡超过 30 分钟——卡住的每一分钟都是可产出时间的浪费。

---

## 简历上的最终描述

四周结束后你的项目可以这样写：

> **Adaptive Research Agent** — 基于 LangGraph 的多智能体协作检索增强生成系统
> - 采用 Supervisor + Researcher + Writer 三代理架构，实现搜索与写作分离
> - Researcher 支持 FAISS 向量检索 + BM25 混合检索 + CrossEncoder 重排 + 上下文压缩
> - 支持父子文档分块、Multi-Query 查询扩展、Tavily 互联网搜索
> - 前后端分离：FastAPI 流式 SSE 接口 + Streamlit 交互界面
> - 支持多轮对话记忆、可配置搜索策略、MCP 外部工具扩展
> - 技术栈：Python, LangGraph, LangChain, FAISS, FastAPI, Streamlit, DeepSeek API
