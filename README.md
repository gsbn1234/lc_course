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

## 架构决策记录（ADR）

每个决策先交代"备选方案是什么、为什么没选"，面试被追问时按这个思路答。

### ADR-1：为什么用 LangGraph，而不是 CrewAI / AutoGen / 手写状态机

**需求驱动**：本系统核心是"Reviewer 审核不过 → 打回 Researcher 带修改意见重搜"的**循环**，
以及**多轮对话持久化**（checkpoint）。需要的是可控制的图结构，不是自由的对话流水线。

- **LangGraph**：`StateGraph` + 条件边 + checkpoint 天然表达"循环直到达标"，能精确控制
  每轮的工具调用和审核回路；`stream_mode` 还支持 `values`（节点状态）+ `messages`（token）双路流式。
- **CrewAI**：偏任务流水线（Role/Task 编排），对复杂循环、checkpoint 记忆的控制较弱。
- **AutoGen**：对话式多智能体，适合"辩论/讨论"型协作；对"检索→写作→审核→打回"这种确定性流程反而不可控。
- **手写状态机**：要自己实现 checkpoint、并发、流式，工程量不划算。

### ADR-2：为什么工具走 MCP 协议，而不是直接把工具 `bind_tools` 给 Researcher

- **隔离性**：MCP server 是独立 stdio 子进程，工具崩溃不会拖垮 Agent 主进程。
- **标准协议**：MCP 是行业标准（Anthropic/OpenAI 均支持），工具可跨语言、跨框架复用，
  不绑死在 LangChain 生态里。
- **领域化白名单**：`mcp_tools/registry.py` 按领域（ai_learning/general/...）过滤要挂载的工具，
  新增领域只加"server 注册 + registry 映射"两处，不改接入链路。
- 代价：stdio 进程间通信多一次 IPC 开销；为复用子进程做了全局客户端 + 锁的单例管理。

### ADR-3：为什么父子两粒度切分（child 搜 / parent 返）

- 小 chunk（200 字符）检索**精度高**，但上下文碎片化；大 chunk（800 字符）上下文**完整**，但召回噪声大。
- 折中：在 child 上建向量 + BM25 索引做精检索，命中后映射回所属 parent 整块喂给 LLM——
  既搜得准，又不丢失上下文。

### ADR-4：为什么前后端分离 + SSE 流式

- Streamlit 直接调 graph 会把 LLM 长时间阻塞在网页会话里，无法逐 token 展示过程。
- 拆成 FastAPI + SSE 后：检索状态、工具调用、Writer token 全部实时推送，前端做打字机效果；
  后端可独立水平扩展，任意前端（网页/移动/CLI）复用同一套 API。

### ADR-5：为什么会话用三层存储（内存 LRU + Redis + 磁盘）

- 最贵的操作是"重载 embedding 模型 + 重建图"（几秒级、烧资源）——内存 LRU 缓存（上限 20 个）
  命中热会话时开销降到零。
- Redis 存会话**元数据**（多进程 / 后端重启共享）；磁盘存**重物**（向量库 / 切块 / bm25 pickle）。
- 冷启动路径：Redis 命中元数据 → 从磁盘重建 → 回填 LRU。重启不丢会话。

---

## 评测与基线对比

同一份 5 题测试集、同一个 LLM 裁判（Context Recall / Faithfulness / Answer Relevancy，0-1 分），
三档系统只换检索链路，打分口径完全一致：

| 档位 | Context Recall | Faithfulness | Answer Relevancy |
|------|---------------|--------------|------------------|
| 无检索直答（LLM 裸答，无上下文） | 0.0000 | 0.0000 | 1.0000 |
| 单路检索（仅向量 top-k） | 0.4800 | 0.8000 | 0.5000 |
| 完整系统（改写+路由+父子+HyDE+精排+压缩） | 0.6600 | 0.8000 | 0.7400 |

- **无检索直答** Recall/Faithfulness 双 0 → 证明"检索增强"的必要性；
- **完整系统** Recall / Answer Relevancy 明显高于单路检索（0.48→0.66、0.50→0.74），
  且只有它能答联网时效题 → 高级检索组件不是炫技（Faithfulness 单次持平，属裁判波动，见下注）；
- 联网题（DeepSeek-R1）只有完整系统能答：router 正确识别时效题 → 走 Tavily → 作答。
  这个缺陷是评估暴露的（原 router 把无关键词的时效题误判为 local），已通过改进 router 提示词修复。

> 局限（面试主动说明）：测试集为自建 5 题小集；裁判为 LLM-as-Judge，单次运行有随机波动，
> 看方向不看绝对值。逐题明细见 `docs/eval_baseline_result.md`。复现：`python eval_baseline.py`。

---

## 性能与成本

> 实测于 2026-08-29，模型 deepseek-v4-flash（DeepSeek 开放平台）。
> 耗时 = 答案阶段墙钟时间；token = 答案阶段 LLM 输入/输出（裁判打分只计入总成本）。
> 一次运行即可复现：`python eval_baseline.py`。

| 档位 | 平均耗时/题 | 平均输入 token | 平均输出 token |
|------|-----------|---------------|---------------|
| 无检索直答 | 3.1s | 223 | 358 |
| 单路检索 | 1.8s | 1,006 | 230 |
| 完整系统 | 35.4s | 3,970 | 2,168 |

- **一次完整三档基线评估（3 档 × 5 题 + 15 次裁判）≈ ¥0.12**：
  78.7k 输入 + 22.1k 输出 token，按官网价折算（输入 ¥1/百万·缓存未命中、输出 ¥2/百万）。
  DeepSeek 已于 2026-08 公告将上调价格，成本需按当时官网价复核。
- **完整系统单题比单路检索慢约 20×**：这是多轮 LLM 调用的代价
  （改写 + 路由 + HyDE + 多查询 + 精排 + 压缩 + Reviewer 审核，必要时打回重搜）；
  换来的回报是 Recall 0.48→0.66，且只有它能答联网时效题。本地检索 + 精排本身 <1s，35s 大头在 LLM 生成。
- 生产环境压延迟的方向：DeepSeek 自动上下文缓存、子查询并行、更小生成模型。

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
├── mcp_tools/                 # 自定义 MCP Server（领域化工具注册）
│   ├── registry.py            #   领域 → 工具白名单（ai_learning/general/...）
│   ├── my_mcp_server.py       #   FastMCP stdio server（暴露全部领域工具）
│   └── tools/
│       └── ai_learning.py     #   AI 学习领域真实工具（GitHub / arXiv / HuggingFace）
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
python eval_baseline.py       # 三档基线对比（无检索 vs 单路 vs 完整）
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
