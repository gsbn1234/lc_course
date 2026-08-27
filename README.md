# lc_course — Multi-Agent 智能检索系统

基于 **LangGraph** 构建的多 Agent 协作检索系统（Supervisor + Researcher + Writer + Reviewer），
集成 **Parent Document 混合检索**（FAISS 向量 + BM25 关键词）、**CrossEncoder 精排**、
**上下文压缩**、**HyDE 假想答案**、**长期记忆** 与 **MCP 工具**，并提供 FastAPI + Streamlit 全栈服务。

> 学习项目：AI Agent 应用开发方向的完整实践（RAG → Agentic RAG → Multi-Agent）。

---

## 核心特性

- **多 Agent 协作**：Supervisor（调度）→ Researcher（本地库 + 联网搜索）→ Writer（成文）→ Reviewer（质量审核，不合格自动打回重搜），各节点各司其职，上下文互不污染
- **混合检索**：child 小块（200 字符）上做 FAISS + BM25 混合，映射回 parent 大块（800 字符），搜得准且上下文完整
- **检索增强**：Query Rewrite、Multi-Query、HyDE、CrossEncoder Rerank、Context Compression 全链路
- **长期记忆**：LangGraph Store 记住用户偏好（"记住…"），SqliteSaver 持久化多轮对话 checkpoint
- **MCP 扩展**：Researcher 可调用自定义 MCP Server 工具
- **流式输出**：SSE（Server-Sent Events）实时推送搜索状态、工具调用和最终答案
- **可评估**：LLM-as-Judge 三维度（Context Recall / Faithfulness / Answer Relevancy）评测脚本

---

## 系统架构

```
                         ┌─────────────┐
                         │   用户提问   │
                         └──────┬──────┘
                                ▼
                        ┌───────────────┐
                        │ rewrite_query │  问题改写 + 写入长期记忆
                        └──────┬────────┘
                               ▼
                         ┌────────────┐
                    ┌───▶│ supervisor │◀───┐
                    │    └─────┬──────┘    │
                    │          │           │
               researcher      │       writer
                    │          │           │
   ┌────────────────┴──┐       │     ┌─────┴──────┐
   │ researcher_agent  │◀──────┘     │ writer_agent│ 生成最终回答
   └────────┬──────────┘             └──────┬─────┘
            │ 工具调用                        │
            ▼                                ▼
   ┌────────────────┐                ┌──────────────┐
   │  local_search  │  FAISS+BM25    │ reviewer_agent│ 质量审核
   │ internet_search│  Tavily 联网    │  忠实/完整/相关│
   │  MCP tools     │  自定义 MCP 工具 └──┬────┬──────┘
   └────────────────┘             PASS │    │ REVISE(未超上限)
                                      ▼    ▼
                                    END  researcher_agent（带修改意见重搜）
```

前端 Streamlit(8501) ↔ HTTP/SSE ↔ FastAPI(8000) ↔ LangGraph Agent-RAG

---

## 技术栈

| 分类 | 技术 |
|------|------|
| 编排 | LangGraph、LangChain |
| 模型 | DeepSeek API（deepseek-v4-flash）、BAAI/bge-small-zh-v1.5（Embedding）、BAAI/bge-reranker-base（Rerank） |
| 检索 | FAISS 向量库、rank_bm25、jieba 中文分词 |
| 联网 | Tavily Search API |
| 服务 | FastAPI + SSE、Streamlit、MCP（FastMCP） |
| 记忆 | LangGraph Store（InMemory）、SqliteSaver（checkpoints.sqlite） |
| 可观测 | LangSmith 全链路追踪（LLM 调用 / Agent 步骤 / 工具轨迹） |
| 测试 | pytest（tests/ 目录，离线测试，不联网不烧 token） |

---

## 目录结构

```
lc_course/
├── main.py                    # 脚本入口：加载 docs → 建索引 → 跑多 Agent 图
├── multi_agent/               # 核心库
│   ├── config.py              #   统一路径与配置（以项目根为基准）
│   ├── loader.py              #   PDF 文档加载
│   ├── parent_splitter.py     #   父子两粒度切分（child 搜 / parent 返回）
│   ├── embedding.py           #   bge 向量化
│   ├── vector_db.py           #   FAISS 建库 / 加载（路径统一走 config）
│   ├── bm25.py                #   BM25 关键词索引
│   ├── retriever.py           #   基础检索函数（vector / bm25）
│   ├── parent_retriever.py    #   父子混合检索（核心检索层）
│   ├── query_rewrite.py       #   查询改写
│   ├── multi_query.py         #   多角度查询生成
│   ├── hyde.py                #   HyDE 假想答案
│   ├── reranker.py            #   CrossEncoder 精排
│   ├── context_compressor.py  #   上下文压缩（逐句打分裁剪）
│   ├── web_retriever.py       #   Tavily 联网检索
│   ├── router_graph.py        #   Query Routing RAG 图（local/web）
│   ├── multi_agent_graph.py   #   多 Agent 协作图（Supervisor/Researcher/Writer/Reviewer）
│   └── llm.py                 #   DeepSeek 实例
├── streamlit_1/
│   ├── backend.py             # FastAPI 后端（上传 / 流式对话 / 健康检查）
│   └── app.py                 # Streamlit 前端
├── mcp/my_mcp_server.py       # 自定义 MCP Server（示例工具）
├── docs/                      # 知识库 PDF（喂给 RAG 的原始文档）
├── evaluate.py                # LLM-as-Judge 三维度评估
├── eval_compare.py            # HyDE 开关对比评估
├── eval_dataset.py            # 评估测试集
├── test_backend.py            # FastAPI 链路冒烟测试（手动脚本，需真后端）
├── pytest.ini                 # pytest 配置（testpaths=tests，只收 tests/）
├── tests/                     # pytest 单元测试（离线，不联网）
│   ├── conftest.py            #   把项目根加入 sys.path
│   ├── test_config.py         #   路径 / 环境变量体检
│   ├── test_retriever.py      #   BM25 / 向量检索 / 父子映射
│   └── test_multi_agent_graph.py # 图结构 + Reviewer 行为 + 答案提取
├── langgraph.json             # langgraph-cli 配置
└── faiss_db/                  # 生成的向量索引（勿提交 git）
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt   # 或 uv sync
```

### 2. 配置环境变量

复制 `.env` 并填入你的 Key：

```ini
DEEPSEEK_API_KEY=你的DeepSeekKey
TAVILY_API_KEY=你的TavilyKey
LANGSMITH_API_KEY=可选
LANGSMITH_TRACING=false
```

### 3. 启用 LangSmith 追踪（可选）

在 `.env` 中把 `LANGSMITH_TRACING` 改为 `true` 并填入真实 `LANGSMITH_API_KEY`
（[smith.langchain.com](https://smith.langchain.com) → Settings → API Keys，`lsv2_` 开头），
`LANGSMITH_PROJECT` 建议改成 `lc-course`。不填 Key 或保持 `false`，程序静默关闭追踪，不影响任何功能。

开启后，每次跑图（`main.py` / 后端流式对话）都会自动上报：每一步 LLM 的输入输出、token 消耗、
Researcher 的工具调用、Reviewer 的审核结论。在 LangSmith 里可按 `run_name`（`multi_agent_rag` / `chat_stream`）筛选。

### 4. 运行方式（三选一）

**方式 A：脚本直接跑（最快）**
```bash
python main.py
```

**方式 B：Web 全栈（FastAPI + Streamlit）**

终端 1 —— 启动后端：
```bash
uvicorn streamlit_1.backend:app --port 8000 --reload
```
终端 2 —— 启动前端：
```bash
streamlit run streamlit_1/app.py
```
浏览器打开 `http://localhost:8501`，上传 PDF → 构建索引 → 提问。

### 5. 跑评估

```bash
python evaluate.py            # 三维度评估
python eval_compare.py        # HyDE 开关对比
```

### 6. 跑单元测试（pytest）

```bash
# 首次先装测试依赖
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 跑全部测试（全部离线：假 LLM + 合成文档，不联网、不烧 token）
.venv\Scripts\python.exe -m pytest
```

测试覆盖：`config` 路径/环境变量、BM25 / 向量检索 / 父子映射、
多 Agent 图结构（节点/边线）、Reviewer 节点行为、答案提取。

> 注：`pytest.ini` 里 `testpaths = tests`，只收集 `tests/` 目录。
> 项目根的 `test_backend.py` 是手动冒烟脚本（需真后端 + 真 API），不会被 pytest 误收集。

---

## 常见问题

- **换目录跑就报错 / 找不到 docs**：所有路径统一在 `multi_agent/config.py` 中按项目根计算，请勿自行硬编码相对路径。
- **faiss_db 索引过期**：`docs/` 内容更新后需删除 `faiss_db/` 重新建索引（目前索引按"存在即复用"策略，暂未自动校验文档变更）。
- **对话记忆重启丢失**：默认 `checkpoints.sqlite` 持久化 checkpoint；长期偏好记忆用的是内存 Store，重启即清空。

---

## 备份说明

清理死代码时删除的旧版本模块已打包至
`D:\python2\lc_course_已删除代码备份_20260822.zip`（含 `legacy/`、`LangChainRAG/` 及误建的残留目录），如需找回可解压恢复。
