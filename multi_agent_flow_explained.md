# Multi-Agent 协作系统 — 完整代码执行流程详解

> 配合你跑 `main.py` 时终端里打印的每一行日志来读。日志走到哪，就看对应的解释。

---

## 先搞清楚：你的系统里有三个"人"

在你原来的 `tool_agent_graph.py` 里，只有一个 Agent——它既要搜索、又要写作，所有事情自己扛。

现在拆成了三个"人"，每个只管一件事：

| 角色 | 干什么 | 有什么工具 | 比喻 |
|------|--------|-----------|------|
| **Supervisor**（调度者） | 看状态，决定"下一步谁干活" | 无，纯逻辑判断 | 项目经理，不写代码但分任务 |
| **Researcher**（研究员） | 搜本地库 + 搜互联网 | `local_search`、`internet_search` | 实习生，跑腿查资料，但不写报告 |
| **Writer**（写作者） | 把研究员搜到的材料写成答案 | 无，纯写作 | 文案，拿别人整理的材料写最终报告 |

---

## 阶段一：main.py 启动 — 准备工作（跑一次）

```python
# main.py 第 15-31 行
```

这一步跟你之前所有版本一模一样，不做任何 Agent 相关的事情：

```
① load_documents()
   → PyPDFDirectoryLoader 读取 D:/claude/Claude_Outputs 下 8 个 PDF
   → 共 22 页，每页是一个 LangChain Document
   → 打印: "加载文档页数：22"

② split_parent_child(docs, child_size=200, parent_size=800, overlap=50)
   → 同一个 PDF 集被切两次：
      大块（parent）：800 字一块 → 39 块，每块有唯一 parent_id
      小块（child）：200 字一块 → 175 块，每块继承对应 parent 的 parent_id
   → 打印: "Parent chunks: 39"、"Child chunks: 175"
   → 返回 (child_docs, parent_docs)

③ get_embeddings()
   → HuggingFaceEmbeddings("BAAI/bge-small-zh-v1.5")
   → 加载 71 个权重文件到内存
   → 打印: "Loading weights: 100% ..."
   → 返回 embeddings 模型

④ get_vector_store(child_docs, embeddings)
   → 把 175 个 child chunk 编码成 384 维向量 → 存入 FAISS 索引
   → 保存到 faiss_db/ 文件夹（下次启动直接用，跳过重建）

⑤ create_bm25(child_docs)
   → 175 个 child chunk → jieba 分词 → BM25Okapi 索引
   → 打印: "BM索引建立完成"

⑥ get_llm()
   → ChatOpenAI(model="deepseek-v4-flash", temperature=0.3)
   → 返回 llm 实例
```

---

## 阶段二：编译图 — 把蓝图搭好（跑一次）

```python
# main.py 第 40-47 行
graph = build_multi_agent_graph(llm, child_vector_store, child_bm25, child_docs, parent_docs, max_tool_rounds=5)
```

这一步跳进 `multi_agent_graph.py` 的 `build_multi_agent_graph` 函数（第 353-427 行）。

### 2.1 制造 Researcher 的工具

```python
# 第 385 行
search_tools = make_search_tools(vector_store, bm25, chunks, parent_docs)
```

跳到 `make_search_tools` 函数（第 69-136 行）。这个函数做了两件事：

**第一件：** 定义 `local_search` 工具（第 76-113 行）。这个工具内部的流程链是：

```
用户调用 local_search(query="RAG原理")
    │
    ├─ ① parent_hybrid_retrieve(query, top_k=5)
    │     ├─ vector_search → FAISS 找 5 个最相似的 child chunk
    │     ├─ bm25_search → BM25 找 5 个关键词最匹配的 child chunk
    │     ├─ 按 parent_id 合并 → 映射到 800 字的 parent 大块
    │     └─ MD5 去重 → 返回 top 5 个 parent
    │
    ├─ ② rerank(query, [5个parent doc])
    │     └─ CrossEncoder 逐对打分 → 取 top-3
    │
    ├─ ③ compress_documents(query, [3条], reranker)
    │     └─ 每条拆成句子 → CrossEncoder 逐句打分 → 保留 top-50% 句子
    │
    └─ ④ 拼成格式化字符串返回给 LLM
        "[文档1 | 来源=vector_databases.pdf | 相关度=0.912]\nRAG是一种..."
```

**第二件：** 定义 `internet_search` 工具（第 115-134 行）。逻辑类似但简单很多——调 `web_search()`（就是 Tavily API），搜 5 条网页结果，格式化成字符串返回。

**第三件：** 把两个工具包装成列表返回（第 136 行）：
```python
return [local_search, internet_search]
```

### 2.2 给 LLM 装上工具

```python
# 第 386-387 行
researcher_llm = llm.bind_tools(search_tools)
researcher_tool_node = ToolNode(search_tools)
```

`llm.bind_tools(search_tools)` 做了什么：把你的 DeepSeek LLM 和两个搜索工具"绑定"。绑定之后，LLM 发出的 API 请求里会附带两个工具的 name、description、参数格式。LLM 看到这些，就知道它可以用 `local_search(query="xxx")` 和 `internet_search(query="xxx")` 来干活——它会在输出里附带 `tool_calls`，告诉系统"我要调这个工具"。

`ToolNode(search_tools)` 是什么：LangGraph 提供的一个内置节点，专门负责解析 LLM 输出的 `tool_calls`，找到对应的工具函数去执行，把执行结果包装成 `ToolMessage` 返回。

这两行形成了 Researcher 的"思考-行动"能力：
- `researcher_llm` → Researcher 的"大脑"（知道有哪些工具可用）
- `researcher_tool_node` → Researcher 的"手"（实际去执行工具调用）

### 2.3 注册节点 + 连线

```python
# 第 389-427 行
```

**注册 5 个节点：**

```
节点名                        函数                      什么时候被调用
────────────────────────────────────────────────────────────────────
rewrite_query_node    →    rewrite_query_node       刚启动，第一个
supervisor_node       →    supervisor_node          rewrite 之后 / Researcher 搜完 / Writer 写完
researcher_agent      →    researcher_agent_node    Supervisor 说"该搜了"
researcher_tool_node  →    researcher_tool_execute  Researcher 说要调工具
writer_agent          →    writer_agent_node        Supervisor 说"该写了"
```

**连线（节点之间的箭头）：**

```
[START]
    │  (无条件)
    ▼
rewrite_query_node
    │  (无条件，第 400 行)
    ▼
supervisor_node
    │
    ├── next_agent="researcher" → researcher_agent      (第 403-411 行，条件边)
    ├── next_agent="writer"     → writer_agent          (第 403-411 行，条件边)
    └── next_agent="finish"     → END                   (第 403-411 行，条件边)
                                    │
researcher_agent                    │
    │                               │
    ├── next_agent="researcher" → researcher_tool_node  (第 414-421 行，条件边)
    └── next_agent="writer"     → supervisor_node       (第 414-421 行，条件边)
                                    │
researcher_tool_node                │
    │  (无条件，第 422 行)           │
    ▼                               │
researcher_agent ← 回到循环入口     │
                                    │
writer_agent                        │
    │  (无条件，第 425 行)           │
    ▼                               │
supervisor_node ─────────────────────┘
```

把上面连起来看：

```
                       ┌─────────────────────────────┐
                       │    Researcher 内部循环        │
                       │                             │
  [START]              │  researcher_agent           │
      │                │      │                      │
      ▼                │      │ next_agent="researcher"
  rewrite_query_node   │      ▼                      │
      │                │  researcher_tool_node       │
      ▼                │      │                      │
  supervisor_node ─────┤      │ 无条件返回            │
      │                │      ▼                      │
      │                │  researcher_agent           │
      │                │      │                      │
      │                │      │ next_agent="writer"   │
      │                │      ▼                      │
      │                └─────→ supervisor_node       │
      │                           │ (循环出口)        │
      │                           └───────────────────┘
      │                     next_agent="writer"
      ▼
  writer_agent
      │
      ▼
  supervisor_node
      │
      │ next_agent="finish"
      ▼
    [END]
```

---

## 阶段三：处理一个问题 — graph.invoke() 后的逐步执行

```python
# main.py 第 61-65 行
result = graph.invoke({
    "question": "什么是RAG？",
    "researcher_rounds": 0,
    "max_tool_rounds": 5,
})
```

意思是：把这个问题扔进编译好的图里，图会自动从 START 开始，沿着边一步一步走，直到走到 END。

### 步骤 1：rewrite_query_node — 改写问题 + 给三个"人"发任务

```
代码位置：multi_agent_graph.py 第 141-176 行
触发方式：START → rewrite_query_node（无条件边）
```

**做了什么事：**

```
用户问题: "什么是RAG？"
    ↓ 调用 rewrite_query(question, llm) → 一次 LLM 调用
    ↓ 输出: "RAG（检索增强生成）的定义、核心原理、系统架构..."
    ↓

给 Researcher 写一份"任务说明书"（SystemMessage）:
  "你是一个专职的信息研究员。
   你的唯一任务：调用搜索工具获取信息。
   搜完以「研究材料」形式整理好，等待 Writer 接手。"
    ↓ 存进 researcher_messages[0]

再给 Researcher 写一份具体的"搜索指令"（HumanMessage）:
  "搜索指令：请针对以下主题进行全面搜索。
   用户原始问题：什么是RAG？
   搜索参考方向：RAG（检索增强生成）的定义、核心原理...
   请先搜索本地知识库，如果不够再搜互联网。"
    ↓ 存进 researcher_messages[1]

Writer 的消息通道先空着:
  writer_messages = []

设置调度信号:
  next_agent = "researcher"  ← 告诉 Supervisor：先去 Researcher
```

**打印：** `===== Query Rewrite =====` + 改写结果

**State 当前值：**
```
rewrite_question = "RAG（检索增强生成）的定义、核心原理、系统架构..."
researcher_messages = [SystemMessage, HumanMessage]
writer_messages = []
next_agent = "researcher"
```

---

### 步骤 2：supervisor_node — Supervisor 第一次看状态

```
代码位置：multi_agent_graph.py 第 179-231 行
触发方式：rewrite_query_node → supervisor_node（第 400 行，无条件边）
```

**做了什么事：**

```
读 state["next_agent"] → "researcher"

进入第一个 if 分支（第 190 行）:
  打印: "[Supervisor] → 派 Researcher 去搜索"
  返回: {"next_agent": "researcher"}  ← 不变，继续派 Researcher
```

**打印：** `[Supervisor] → 派 Researcher 去搜索`

---

### 步骤 3：route_supervisor — 条件判断，去 researcher_agent

```
代码位置：multi_agent_graph.py 第 297-305 行
触发方式：supervisor_node 之后自动调用
```

**做了什么事：**

```
读 state["next_agent"] → "researcher"
匹配 elif → 返回 "researcher_agent"  ← 告诉图：去 researcher_agent 节点
```

---

### 步骤 4：researcher_agent_node — Researcher 第一次思考

```
代码位置：multi_agent_graph.py 第 234-278 行
触发方式：route_supervisor 返回 "researcher_agent"
```

**做了什么事：**

```
① 读 researcher_messages:
   [0] SystemMessage("你是专职研究员...")
   [1] HumanMessage("搜索指令：请针对以下主题进行全面搜索...")

② researcher_llm.invoke(researcher_messages)
   → LLM 读到 SystemMessage: "你是研究员，只管搜索"
   → LLM 读到 HumanMessage: "用户问什么是RAG，搜索参考方向是..."
   → LLM 自己分析: "RAG 是技术概念 → 先搜本地库"
                    "改写后的查询很长 → 拆成多个角度分别搜"

③ LLM 输出的 AIMessage:
   内容 = ""  (空，因为只是工具调用)
   tool_calls = [
     {"name": "local_search", "args": {"query": "RAG 检索增强生成 定义 核心原理"}},
     {"name": "local_search", "args": {"query": "RAG 系统架构 检索器 生成器 工作流程"}},
     {"name": "internet_search", "args": {"query": "RAG 定义 原理 架构 应用场景"}},
     {"name": "internet_search", "args": {"query": "Retrieval-Augmented Generation RAG architecture..."}},
   ]
   ← 两个 local_search + 两个 internet_search 一次性并行发出！

④ 检查 tool_calls 是否存在 → 存在，说明 Researcher 还要搜
   扣 rounds = 0, max_rounds = 5 → 0 < 5，没超限

⑤ 返回:
   researcher_messages: [AIMessage(tool_calls)]  ← 追加到 messages 列表后面
   researcher_rounds: 1                           ← 从 0 变成 1
   next_agent: "researcher"                       ← 保持，还要继续搜
```

**此时 researcher_messages 变成了：**
```
[0] SystemMessage("你是专职研究员...")
[1] HumanMessage("搜索指令：...")
[2] AIMessage(tool_calls=[4个工具调用])  ← 新加的
```

---

### 步骤 5：route_researcher — 搜索循环的分岔口

```
代码位置：multi_agent_graph.py 第 308-318 行
触发方式：researcher_agent_node 之后自动调用
```

**做了什么事：**

```
读 state["next_agent"] → "researcher"  ← 上一步设的
匹配 if → 返回 "researcher_tool_node"   ← 还有工具要执行，先去干活
```

---

### 步骤 6：researcher_tool_execute — 实际执行搜索工具

```
代码位置：multi_agent_graph.py 第 321-348 行
触发方式：route_researcher 返回 "researcher_tool_node"
```

**做了什么事：**

```
① 读 researcher_messages[-1] → 上一步的 AIMessage(tool_calls)

② 遍历 tool_calls，打印日志:
   [Researcher] local_search({'query': 'RAG 检索增强生成 定义 核心原理'})
   [Researcher] local_search({'query': 'RAG 系统架构 检索器 生成器 工作流程'})
   [Researcher] internet_search({'query': 'RAG 定义 原理 架构 应用场景'})
   [Researcher] internet_search({'query': 'Retrieval-Augmented Generation RAG architecture...'})

③ researcher_tool_node.invoke({"messages": research_messages})
    ↓ ToolNode 内部工作:
       解析 tool_calls → 找到对应的函数
       → 执行 local_search("RAG 检索增强生成 定义 核心原理")
          ↓ 内部链路: parent_hybrid_retrieve → rerank → compress → 格式化
          → 返回字符串: "[文档1 | 来源=rag_advanced.pdf | 相关度=0.912]\nRAG是一种..."
       → 执行 local_search("RAG 系统架构...")
          → 同上
       → 执行 internet_search("RAG 定义 原理...")
          → 调 Tavily API → 返回 5 条网页结果 → 格式化
       → 执行 internet_search("Retrieval-Augmented Generation...")
          → 同上
       → 最终返回 4 条 ToolMessage（每条包含搜索结果的文本）

④ 返回:
   researcher_messages: [ToolMessage×4]  ← 4 条搜索结果追加到消息列表
   next_agent: "researcher"             ← 回到 Researcher 让它看结果
```

**此时 researcher_messages 变成了：**
```
[0] SystemMessage("你是专职研究员...")
[1] HumanMessage("搜索指令：...")
[2] AIMessage(tool_calls=[4个工具调用])
[3] ToolMessage(local_search 结果1: "RAG是一种检索增强生成技术...")
[4] ToolMessage(local_search 结果2: "RAG的工作流程包含...")
[5] ToolMessage(internet_search 结果1: "RAG（Retrieval-Augmented Generation）...")
[6] ToolMessage(internet_search 结果2: "Retrieval-Augmented Generation is...")
```

---

### 步骤 7-8：Researcher 内部再循环

```
researcher_tool_node → (无条件边) → researcher_agent_node（步骤 4 再来一次，看结果）
```

Researcher 的第二轮思考：

```
① LLM 重新读 researcher_messages
   现在有 System + Human + 自己的 tool_calls + 4 条搜索结果

② LLM 读 4 条搜索结果后判断:
   "本地库覆盖了 RAG 定义、架构、流程这些基础内容。
    但还缺少 RAG 的演进历史和不同的范式类型。
    互联网上也有更多补充信息。
    再搜一轮。"

③ LLM 输出新的 AIMessage:
   tool_calls = [
     {"name": "internet_search", "args": {"query": "RAG 起源 论文 2020 Lewis..."}},
     {"name": "internet_search", "args": {"query": "Naive RAG Advanced RAG..."}},
     {"name": "local_search", "args": {"query": "RAG 幻觉 向量数据库 知识库..."}},
   ]

④ 还有 tool_calls，rounds=1 < 5，继续

⑤ 返回 → route_researcher → researcher_tool_execute
   → 执行 3 个工具 → 回到 researcher_agent_node 再看结果

...（循环，直到 Researcher 觉得搜够了或达到 5 轮上限）
```

---

### 步骤 9：Researcher 决定收手 — 最后一轮

```
代码位置：researcher_agent_node 第 273-278 行（正常）或第 248-267 行（超限被迫）
```

**情况 A：正常收手（LLM 自己觉得够了）**

```
LLM 输出 AIMessage:
  content = "搜索完毕，以下是研究材料：\n\n## 本地库搜索结果\nRAG是一种检索增强生成...\n## 互联网搜索结果\nRAG最早由Lewis等人在2020年提出..."
  tool_calls = []  ← 空！不再要求调工具

第 246 行：getattr(response, "tool_calls", None) → None
进入 else 分支（第 273 行）→ 返回:
  researcher_messages: [AIMessage(content="搜索完毕...")]
  researcher_rounds: 2（假设搜了两轮）
  next_agent: "writer"   ← 切换到 Writer
```

**情况 B：被迫收手（搜了 5 轮还没收手）**

```
LLM 输出 AIMessage:
  tool_calls = [{"name": "internet_search", ...}]  ← 还要搜

第 246 行：tool_calls 不为空
第 247 行：max_rounds = 5
第 248 行：rounds = 5  → 5 >= 5 → 进入超限分支

① 构造一条 summary_prompt:
   "已达到最大搜索轮数，请不要再搜索。
    立即将你已获得的所有搜索结果整理成一份结构化的研究材料。"

② 创建一个"禁用工具"的 LLM:
   summary_llm = researcher_llm.bind_tools([])
   ← 绑定空工具列表，LLM 没法再输出 tool_calls

③ 调用一次不带工具的 LLM:
   final_response = summary_llm.invoke(messages + [response, summary_prompt])
   → LLM 收到指令"不准搜了，总结你手上的材料"
   → 输出 AIMessage(content="搜索材料整理如下：\n\n## 本地库结果\n...\n## 互联网结果\n...")

④ 返回:
   researcher_messages: [response, summary_prompt, final_response]
   ← 三条一起追加：LLM 的 tool_calls + 强制要求总结的指令 + 最终的总结材料
   researcher_rounds: rounds + 1
   next_agent: "writer"
```

---

### 步骤 10：route_researcher — 这次走另一条路

```
读 state["next_agent"] → "writer"
不匹配 if → else → 返回 "supervisor_node"
```

这一次不再是回 `researcher_tool_node` 继续循环，而是回去找 Supervisor 了。

---

### 步骤 11：supervisor_node — Supervisor 第二次调度（关键转折）

```
代码位置：multi_agent_graph.py 第 194-227 行
触发方式：route_researcher 返回 "supervisor_node"
```

**做了什么事：**

```
读 state["next_agent"] → "writer"  ← 第一次来这里时是 "researcher"，现在是 "writer"

进入 elif 分支（第 194 行）:
  打印: "[Supervisor] → Researcher 搜完了，派 Writer 来写"

① 从 researcher_messages 里提取 Research 的最终成果:
   research_messages = [System, Human, AIMessage, ToolMessage×7, AIMessage(content="搜索完毕，研究材料如下：...")]
   
   从后往前找第一条有 content 的 AIMessage（第 201-204 行）:
   倒数第 1 条: AIMessage(content="搜索完毕，研究材料如下：\n\n## 本地库搜索结果\nRAG是一种...\n## 互联网结果\n...", tool_calls=[])
   → content 非空 → 这就是 research_content！

② 给 Writer 写一份"任务说明书"（SystemMessage）:
   "你是一个专业的技术文档写作者。
   你的任务：基于 Researcher 提供的研究材料，撰写清晰、准确、结构化的回答。
   所有事实必须来自研究材料，不要编造。"

③ 把 Researcher 的研究材料 + 用户问题打包成 HumanMessage:
   "用户问题：什么是RAG？
    研究材料：[上面提取到的 research_content]
    请基于以上材料撰写最终回答。"

④ 返回:
   writer_messages: [writer_system, writer_human]
   research_results: research_content
   next_agent: "writer"
```

**打印：** `[Supervisor] → Researcher 搜完了，派 Writer 来写`

---

### 步骤 12：route_supervisor → writer_agent

```
读 state["next_agent"] → "writer"
elif 匹配 → 返回 "writer_agent"
```

---

### 步骤 13：writer_agent_node — Writer 写最终回答

```
代码位置：multi_agent_graph.py 第 280-292 行
触发方式：route_supervisor 返回 "writer_agent"
```

**做了什么事：**

```
① 读 writer_messages:
   [0] SystemMessage("你是专业的技术文档写作者...")
   [1] HumanMessage("用户问题：什么是RAG？\n研究材料：...\n请撰写最终回答。")

② llm.invoke(messages)
   ← 注意：这里的 llm 是没有 bind_tools 的原始 llm！
   ← Writer 不需要任何工具，只需要写作能力
   → LLM 读 SystemMessage: "你是一个写作者，基于研究材料写回答"
   → LLM 读 HumanMessage: "用户问题...研究材料...请撰写"
   → LLM 输出: "# 什么是RAG？\n\nRAG（检索增强生成）是一种将信息检索与大语言模型生成相结合的技术框架..."
   ← 这是用户最终看到的答案

③ 返回:
   writer_messages: [AIMessage(content="# 什么是RAG？\n\nRAG是一种...")]
   next_agent: "finish"
```

---

### 步骤 14：writer_agent → supervisor_node → END

```
writer_agent → (第 425 行，无条件边) → supervisor_node

supervisor_node:
  读 next_agent → "finish"
  进入 else 分支（第 229 行）
  打印: "[Supervisor] → 任务完成"
  返回: {"next_agent": "finish"}

route_supervisor:
  读 next_agent → "finish"
  不匹配 if，不匹配 elif → else → 返回 END
```

---

### 步骤 15：回到 main.py

```python
# main.py 第 67-69 行
answer = extract_answer(result)
print(f"\n回答:\n{answer}")
print(f"\nResearcher 搜索轮数: {result.get('researcher_rounds', 0)}")
```

`extract_answer`（第 432-438 行）从 `writer_messages` 里找到 Writer 写的最终回答，返回纯文本。

---

## 完整流程图

```
main.py 启动
  │
  ├── 初始化（一次，阶段一）：PDF → 父子切分 → FAISS + BM25 → LLM
  │
  └── build_multi_agent_graph()（一次，阶段二）
        │
        │  ① make_search_tools → local_search + internet_search
        │  ② llm.bind_tools → researcher_llm（Researcher 专用大脑）
        │  ③ ToolNode → researcher_tool_node（Researcher 的双手）
        │  ④ 注册 5 节点 + 连边 + compile
        │
        ▼  graph.invoke({"question": "什么是RAG？"})（每次一个问题的流程，阶段三）
        │
  ① rewrite_query_node        改问题 → 给 Researcher 写任务书 → 初始化消息
        │
  ② supervisor_node           看状态 → "researcher" → 派 Researcher
        │
  ③ route_supervisor          → "researcher_agent"
        │
  ┌─────────────────────────────────────────────────────────────┐
  │               Researcher 内部 ReAct 循环                     │
  │                                                             │
  │  ④ researcher_agent_node                                   │
  │     读 researcher_messages                                  │
  │     → researcher_llm.invoke → 输出 AIMessage(tool_calls)    │
  │     → 检查 tool_calls 是否存在                              │
  │       有 → 检查 rounds 是否超限                              │
  │            没超限 → next_agent="researcher" 继续            │
  │            超限 → bind_tools([]) 强制总结 → next_agent="writer"│
  │       没有 → 自然收手 → next_agent="writer"                 │
  │        │                                                    │
  │  ⑤ route_researcher                                        │
  │     next_agent="researcher" → "researcher_tool_node"        │
  │     next_agent="writer"     → "supervisor_node"（循环出口）  │
  │        │ (情况A: 继续)                                       │
  │  ⑥ researcher_tool_execute                                  │
  │     → 打印工具调用日志                                       │
  │     → ToolNode 执行 local_search / internet_search          │
  │       每个 local_search 内部:                                │
  │         parent_hybrid_retrieve → rerank → compress → 格式化  │
  │       每个 internet_search 内部:                              │
  │         Tavily API → 格式化                                  │
  │     → 返回 ToolMessage                                      │
  │        │                                                    │
  │     → 无条件回到 ④（循环）                                    │
  └─────────────────────────────────────────────────────────────┘
        │ (情况B: 搜完)
        ▼
  ⑦ supervisor_node           搜完了 → 从 researcher_messages 提取研究材料
        │                      给 Writer 写任务书 + 塞材料
        │                      next_agent="writer"
        │
  ⑧ route_supervisor          → "writer_agent"
        ▼
  ⑨ writer_agent_node         读 writer_messages → llm.invoke
        │                      → AIMessage(content="# 什么是RAG？...")
        │                      next_agent="finish"
        ▼
  ⑩ supervisor_node           next_agent="finish"
        │
  ⑪ route_supervisor          → END
        │
        ▼
  main.py  extract_answer → print 最终答案
```

---

## 关键的 State 字段 — 三个"人"各自看什么、写什么

| 字段 | 谁写 | 谁读 | 干什么用 |
|------|------|------|---------|
| `question` | main.py（invoke 时） | rewrite_query_node、supervisor_node | 用户原始问题 |
| `rewrite_question` | rewrite_query_node | 你目前没其他地方读它，但可以给 Researcher 做搜索参考 | 改写后的问题 |
| `researcher_messages` | rewrite_query_node（初始化）、researcher_agent_node（追加 AIMessage）、researcher_tool_execute（追加 ToolMessage） | researcher_agent_node（LLM 看）、supervisor_node（提取研究材料） | Researcher 的独立消息通道 |
| `writer_messages` | supervisor_node（初始化） | writer_agent_node（LLM 看）、extract_answer（取答案） | Writer 的独立消息通道 |
| `research_results` | supervisor_node（从 researcher_messages 提取后存） | 目前没读，但留着备用 | Researcher 的成果副本 |
| `next_agent` | 几乎所有节点都写 | supervisor_node、route_supervisor、route_researcher | 整个系统的"方向盘"，决定下一个节点去哪 |
| `researcher_rounds` | researcher_agent_node | researcher_agent_node | 记数，防止无限循环 |
| `max_tool_rounds` | main.py（invoke 时） | researcher_agent_node | 上限值 |

---

## 为什么拆成三个 Agent，而不是一个 Agent 全包

| 方面 | 一个 Agent（tool_agent_graph） | 三个 Agent（multi_agent_graph） |
|------|-------------------------------|-------------------------------|
| LLM 看的上下文 | 所有搜索结果的 raw text 全塞进同一个 messages 列表 | Researcher 看搜索结果，Writer 只看整理好的材料——各看各的，不互相干扰 |
| 职责边界 | LLM 一边搜一边想怎么写，容易在搜索时就开始组织文笔 | 搜索和写作彻底分离，Researcher 不知道怎么写，Writer 不知道怎么搜 |
| 扩展性 | 加一个新角色 → 改 System Prompt + 改图结构 | 加一个 Reviewer Agent → 加节点 + 加一条条件边 |
| 调试难度 | 一个 messages 里混了搜索指令 + 搜索结果 + 写作思路，排查困难 | 每个 Agent 的 messages 独立，排查时只看一个人的 |

---

## 对照终端日志看一遍（用你刚才跑的"什么是RAG？"为例）

```
===== Query Rewrite =====                                         ← 步骤 1
RAG（检索增强生成）的定义、核心原理、系统架构...

[Supervisor] → 派 Researcher 去搜索                                 ← 步骤 2

[Researcher] local_search({'query': 'RAG 检索增强生成 定义...'})     ← 步骤 6（第一次）
[Researcher] local_search({'query': 'RAG 系统架构 检索器...'})
[Researcher] internet_search({'query': 'RAG 检索增强生成 定义...'})
[Researcher] internet_search({'query': 'Retrieval-Augmented...'})

[Parent] 7 parents → top5                                          ← 工具内部的 parent_hybrid_retrieve
[Rerank] top-3 scores: 0.99, 0.82, 0.32                           ← 工具内部的 rerank
[Compress] 3 docs compressed                                       ← 工具内部的 compress

[Researcher] internet_search({'query': 'RAG 起源 论文...'})         ← 步骤 7-8（第二轮）
[Researcher] internet_search({'query': 'Naive RAG Advanced...'})
[Researcher] local_search({'query': 'RAG 幻觉 向量数据库...'})

[Supervisor] → Researcher 搜完了，派 Writer 来写                     ← 步骤 11
[Supervisor] → 任务完成                                            ← 步骤 14

回答:                                                              ← 步骤 15
RAG（Retrieval-Augmented Generation，检索增强生成）是一种...
```
