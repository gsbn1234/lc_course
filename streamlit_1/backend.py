"""
FastAPI 后端 — 把 Multi-Agent 系统封装成 REST API。

三个接口：
  POST /api/upload-pdf    上传 PDF，建索引，返回 session_id
  POST /api/chat-stream   流式对话（SSE），实时推送搜索状态和最终答案
  GET  /api/health        健康检查

架构关系：
  Streamlit 前端 ↔ FastAPI 后端 ↔ LangGraph Agent-RAG 核心
"""

import asyncio #事件循环里不能跑同步阻塞活：用 asyncio.to_thread 把它丢进线程池，期间事件循环继续处理别的请求
import os
import shutil #用于递归清理临时目录（try/finally 保证出错也会删干净）

import uuid #生成唯一会话 id，每个上传 PDF 的用户拥有独立知识库
import json #序列化字典字符串，SSE 传输的数据必须为 JSON 字符串
import tempfile #创建操作系统临时文件夹，接收上传 PDF、加载文档，用完立刻清理，不占用磁盘
from fastapi import FastAPI, UploadFile, File, Form, HTTPException #`UploadFile`：FastAPI 封装的上传文件对象，包含文件名、二进制内容。`File`：用来声明接口参数是上传文件；`Form`：接收非文件表单字段（如领域）
from fastapi.responses import StreamingResponse #StreamingResponse:返回流式响应，适配 SSE 长连接，可以循环 yield 不断向前端发送消息，适合 LLM 流式输出、Agent 运行日志推送
from fastapi.middleware.cors import CORSMiddleware #CORS 跨域中间件.浏览器同源策略：网页域名、端口和后端不一致就会拦截请求；你的 Streamlit 默认端口 8501，FastAPI 端口 8000，端口不同属于跨域，必须开启 CORS。
from pydantic import BaseModel, field_validator #Pydantic 数据校验模型；FastAPI 依靠 BaseModel 自动校验前端传参类型、做参数解析。field_validator：给单个字段挂自定义校验规则
from langchain_community.document_loaders import PyPDFLoader

# 先于任何 multi_agent 模块导入配置日志：config.py 等 import 时就会打日志
from multi_agent.logging_setup import setup_logging
setup_logging()
import logging

logger = logging.getLogger(__name__)

from multi_agent.parent_splitter import split_parent_child
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
from multi_agent.multi_agent_graph import build_multi_agent_graph, extract_answer, load_mcp_tools

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # 全局唯一长期记忆，跨会话共享（内存版，后端重启即清空）
# ========== 持久化记忆：SqliteSaver ==========
# 把对话记忆（checkpoint）存到本地 checkpoints.sqlite 文件，后端重启不丢
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _saver_cm = SqliteSaver.from_conn_string("checkpoints.sqlite")#调用静态方法 `from_conn_string()。`"checkpoints.sqlite"`：SQLite 数据库文件名。
    #文件不存在：自动新建 `checkpoints.sqlite`。文件已存在：打开现有数据库，读取之前保存过的会话记忆
    # 新版 from_conn_string 返回 context manager，手动 __enter__ 让连接保持到进程结束；
    # 旧版直接返回 saver 实例，直接用即可
    checkpointer = _saver_cm.__enter__() if hasattr(_saver_cm, "__enter__") else _saver_cm# #`hasattr(对象,属性名)`：判断对象有没有 `__enter__` 方法。
except ImportError:
    checkpointer = None   # 没装 langgraph-checkpoint-sqlite 就退回内存版，不报错


app = FastAPI(title="Adaptive Research Agent API")


# 应用关停时关闭全局 MCP 子进程，干净退出（不关的话挂在 asyncio 清理上会报噪音错误）
@app.on_event("shutdown")
async def _shutdown_mcp():
    from multi_agent.multi_agent_graph import close_mcp
    await close_mcp()

# CORS：允许 Streamlit 前端（8501 端口）跨域调用后端（8000 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    # 只放行 Streamlit 前端来源。注意：allow_origins=["*"] 与 allow_credentials=True
    # 是浏览器禁止的组合（通配符不可带凭证），必须写死具体 origin。
    allow_credentials=True,#允许携带 cookie 凭证
    allow_methods=["*"],#允许全部 HTTP 方法 GET/POST/PUT/DELETE
    allow_headers=["*"],#允许前端所有请求头
)

# ========== 会话管理 ==========
# 每个用户（session_id）对应一套独立的知识库和 graph
# 生产环境用 Redis —— session_store 把重物落盘 + 元数据进 Redis，重启不丢、多进程共享。
# 内存里只留最近用过的 graph 缓存（session_store 内部 LRU，最多 20 个）。
from streamlit_1.session_store import (
    save_session, get_session, delete_session, active_session_count,
)


# ========== 请求/响应模型 ==========

class ChatRequest(BaseModel):
    session_id: str
    question: str
    max_tool_rounds: int = 5
    # 长期记忆的隔离键：图里用它做 Store 命名空间 ("users", user_id)，
    # 不同 user_id 的偏好互不可见（multi_agent_graph.py 里读、写各一处）。
    # 故意不给默认值：一旦有默认值，"忘了传"的调用方就会悄悄退回共用命名空间，
    # 也就是之前写死 "21702" 那个「所有人共享一份记忆」的老 bug。
    user_id: str

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, v: str) -> str:
        # 这个值会被直接当成 Store 的命名空间键：留着首尾空格会造出
        # "alice" 和 "alice " 两个看起来一样、实际互不相通的用户。
        v = v.strip()
        if not v:
            raise ValueError("user_id 不能为空")
        if len(v) > 64:
            raise ValueError("user_id 不能超过 64 个字符")
        return v

    #FastAPI 接收 POST 请求 json 体依靠`BaseModel`
    #如果前端传参类型错误，FastAPI 自动返回报错，不需要手写 if 判断参数类型


# ========== 上传安全限制 ==========
MAX_FILES = 5                      # 单次最多上传文件数
MAX_FILE_SIZE = 20 * 1024 * 1024   # 单个文件最大 20MB
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 单次上传总量上限 50MB


# ========== 接口 1：上传 PDF + 建索引 ==========

@app.post("/api/upload-pdf")
async def upload_pdf(
    files: list[UploadFile] = File(...),
    domain: str = Form("general"),
):
    """
    上传一个或多个 PDF 文件，自动切分、建向量库 + BM25、编译 Agent 图。
    domain：知识库领域（ai_learning/general/...），决定给 Researcher 挂哪些 MCP 工具。
    返回 session_id，之后的对话接口需要带上这个 ID。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个 PDF 文件")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_FILES} 个文件")

    # 1. 读取并校验文件（类型 / 大小 / 数量），全部在内存中完成，未通过校验的文件不落盘
    file_blobs = []          # [(安全文件名, 二进制内容), ...]
    total_size = 0
    for f in files:
        # 防路径穿越：只取文件名最后一段，剥掉任何目录部分（/ 和 \ 都处理）。
        # 否则恶意文件名如 "../../../etc/x" 会写出临时目录。
        filename = f.filename.replace("\\", "/").split("/")[-1] if f.filename else "upload.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"仅支持 PDF 文件：{filename}")

        content = await f.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"文件为空：{filename}")
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"文件超过 {MAX_FILE_SIZE // (1024*1024)}MB 上限：{filename}")
        total_size += len(content)
        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(status_code=413, detail="上传总量超过上限")

        file_blobs.append((filename, content))

    # 2. 生成会话 ID
    session_id = str(uuid.uuid4())[:8]

    # 3+4. 落盘临时文件 + 解析 PDF。
    #      写盘和 PyPDF 解析都是同步阻塞（磁盘 IO + CPU），直接写在 async 函数里
    #      会把事件循环占死——上传期间别人的 /api/health、对话请求全部排队。
    #      所以整段封成同步函数丢进线程池，事件循环立刻空出来接别的请求。
    #      try/finally 保证临时目录必删，不残留垃圾文件。
    def _save_and_parse(blobs):
        temp_dir = tempfile.mkdtemp()
        try:
            saved_paths = []
            for filename, content in blobs:
                # 用 uuid 前缀重命名，彻底避免同名文件互相覆盖
                safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
                file_path = os.path.join(temp_dir, safe_name)
                with open(file_path, "wb") as f_out:
                    f_out.write(content)
                saved_paths.append(file_path)

            parsed = []
            for file_path in saved_paths:
                parsed.extend(PyPDFLoader(file_path).load())
            return parsed
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    docs = await asyncio.to_thread(_save_and_parse, file_blobs)

    try:
        # 5+6. 切分 + 建索引：这是全流程最重的同步部分——
        #      jieba 分词是纯 CPU（首次调用还要加载词典），
        #      FAISS.from_documents 更重：要为每个 chunk 同步发一次 embedding 网络请求。
        #      整段一起丢线程池，不切碎（避免多次线程切换的开销）。
        def _build_index():
            child, parent = split_parent_child(
                docs, child_size=200, parent_size=800, overlap=50
            )
            embs = get_embeddings()
            # use_cache=False：上传必须基于新文档强制重建索引，
            # 不能复用全局 faiss_db/ 里的旧索引（否则每个会话共享同一份旧库）
            vs = get_vector_store(child, embs, use_cache=False)
            return child, parent, vs, create_bm25(child)

        child_docs, parent_docs, vector_store, bm25 = await asyncio.to_thread(_build_index)

        # 7. 构建 Agent 图
        #    这一步留在线程池外：build_multi_agent_graph 只编译图结构（建节点/连边），
        #    不做 IO；真正耗时的 MCP 工具加载本身已经是异步的（await load_mcp_tools）。
        llm = get_llm()
        mcp_tools = await load_mcp_tools(domain)  # 按领域加载 MCP 工具（全局复用客户端）
        graph = build_multi_agent_graph(
            llm, vector_store, bm25, child_docs, parent_docs, max_tool_rounds=5,
            checkpointer=checkpointer,   # ← 持久化记忆；None 时内部自动退回 MemorySaver
            store=store,
            extra_tools=mcp_tools,  # ← 新增：MCP 工具接进 Researcher
        )
    except Exception as e:
        # 索引构建失败：返回 500 + 错误详情。
        # 之前直接抛 500 会让前端收到"没有 session_id 的 200"，前端会 KeyError。
        logger.exception("索引构建失败：%s: %s", type(e).__name__, e)
        raise HTTPException(status_code=500, detail=f"索引构建失败：{type(e).__name__}: {e}")

    # 8. 存进会话：重物落盘 + 元数据进 Redis + 内存缓存（三层，见 session_store）
    #    save_session 是同步函数（把 FAISS 索引 + 两个 JSON 写磁盘，再发 Redis 请求），
    #    大索引落盘是实打实的磁盘 IO，同样丢线程池，不占事件循环。
    await asyncio.to_thread(
        save_session,
        session_id,
        {
            "vector_store": vector_store,
            "bm25": bm25,
            "chunks": child_docs,
            "parent_docs": parent_docs,
            "graph": graph,
            "llm": llm,
            "domain": domain,   # 存领域，会话恢复时按它重新加载对应 MCP 工具
        },
        len(docs),
    )

    return {
        "session_id": session_id,
        "page_count": len(docs),
        "child_chunks": len(child_docs),
        "parent_chunks": len(parent_docs),
        "domain": domain,
    }


# ========== 接口 2：流式对话（核心） ==========

@app.post("/api/chat-stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口，返回 SSE（Server-Sent Events）。

    每条事件的格式：
      {"type": "status", "agent": "researcher", "rounds": 1}
      {"type": "status", "agent": "writer"}
      {"type": "tool_call", "name": "local_search", "args": {...}}
      {"type": "done", "answer": "最终回答..."}
    """
    #Pydantic 模型 — ChatRequest 类（第 49-51 行）自动校验请求体。如果前端少传了 session_id 或类型不对，FastAPI 自动返回 422 错误，不需要手写校验逻辑。
    if not req.question or len(req.question) > 2000:
        raise HTTPException(status_code=400, detail="问题为空或超过 2000 字")
    # 从 Redis 取会话：内存缓存没命中 → 按 Redis 元数据从磁盘重建（async）
    session = await get_session(req.session_id, checkpointer=checkpointer, store=store)
    if not session:
        # 会话不存在返回 404（之前是 200 + {"error": ...}，前端会误判为成功）
        raise HTTPException(status_code=404, detail=f"会话 {req.session_id} 不存在，请先上传 PDF")

    graph = session["graph"]

    async def event_stream():
        result = None
        # 两个记忆维度，别搞混：
        #   thread_id = 会话 ID → 多轮对话记忆（同一 session 的多次提问串起来）
        #   user_id   = 长期记忆命名空间 → 跨会话的偏好记忆（"记住xxx"存这里）
        # user_id 由请求方提供，不再写死：写死等于所有用户共用一份偏好。
        config = {
            "configurable": {"thread_id": req.session_id, "user_id": req.user_id},
            # 给 LangSmith 用的轨迹标识：后端每轮对话生成独立 run，
            # 按 run_name=chat_stream + tag=web 就能在 smith 里筛出线上请求。
            "run_name": "chat_stream",
            "tags": ["web", "sse"],
            "metadata": {"session_id": req.session_id},
        }
        try:
            async for mode, chunk in graph.astream(
                {"question": req.question, "researcher_rounds": 0, "max_tool_rounds": req.max_tool_rounds},
                config=config,
                stream_mode=["values", "messages"],  # values=节点级状态；messages=Writer 生成 token
            ):
                if mode == "values":
                    result = chunk
                    next_agent = chunk.get("next_agent", "")
                    rounds = chunk.get("researcher_rounds", 0)

                    if next_agent == "researcher":
                        yield f"data: {json.dumps({'type': 'status', 'agent': 'researcher', 'rounds': rounds}, ensure_ascii=False)}\n\n"
                    elif next_agent == "writer":
                        # Writer 每轮开写前发一个信号，前端用它清空上一轮草稿
                        yield f"data: {json.dumps({'type': 'writer_start'}, ensure_ascii=False)}\n\n"

                    # 推送工具调用事件
                    research_msgs = chunk.get("researcher_messages", [])
                    if research_msgs:
                        last_msg = research_msgs[-1]
                        tool_calls = getattr(last_msg, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name'], 'args': tc['args']}, ensure_ascii=False)}\n\n"

                elif mode == "messages":
                    # messages 流：每个 chunk 是 (AIMessageChunk, metadata)
                    # 只透传 Writer 的生成 token；Researcher/Reviewer 的调用被过滤，不打扰前端
                    msg_chunk, metadata = chunk
                    if metadata.get("langgraph_node") == "writer_agent":
                        content = getattr(msg_chunk, "content", "")
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

            # 推送最终答案（终稿，含 Reviewer 修订后的结果）
            answer = extract_answer(result)
            yield f"data: {json.dumps({'type': 'done', 'answer': answer}, ensure_ascii=False)}\n\n"#`ensure_ascii=False`：让中文正常显示，不会被转成 `\uXXXX`
        except Exception as e:
            # 关键修复：Agent 中途出错不再直接断连，而是把错误作为 SSE 事件发回前端
            err_msg = f"{type(e).__name__}: {e}"
            logger.exception("Agent 运行出错：%s", err_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': err_msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
#`media_type="text/event-stream"` 声明响应体为 SSE 流式协议，浏览器识别之后开启长连接，持续监听后端推送事件


# ========== 接口 3：健康检查 ==========

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": active_session_count(),
    }
