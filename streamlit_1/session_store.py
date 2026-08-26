"""
Redis 会话存储 — 替代 backend.py 里的内存 sessions dict。

为什么换：
  内存 dict 有两个硬伤——① 后端重启全丢（用户要重新传 PDF）；
  ② uvicorn --workers N 起多进程时，每个进程各有一份内存，会话互相看不见。
  Redis 是独立服务，重启、多进程都能读到同一份数据。

三层存储分工：
  Redis   → 会话元数据：session_id → 索引路径、页数、创建时间
  磁盘    → 每个会话的重物：FAISS 索引目录 + chunks/parent_docs/bm25 的 pickle
  内存缓存→ 最近用过的 graph：重建一次后缓存，避免每次请求都重载模型

目录结构：
  faiss_db/sessions/{session_id}/
    ├── faiss_index/       # FAISS 向量库（save_local 产出）
    ├── chunks.pkl         # child Document 列表
    ├── parent_docs.pkl    # parent Document 列表
    └── bm25.pkl           # BM25 索引

★ pickle 安全提醒：pickle.load 对不可信数据有代码执行风险。
  本项目只加载自己存的会话（本地学习用），可接受；生产环境应改 JSON 或加签名校验。
"""
import os
import pickle
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import redis
from langchain_community.vectorstores import FAISS

from multi_agent.bm25 import create_bm25
from multi_agent.embedding import get_embeddings
from multi_agent.llm import get_llm
from multi_agent.multi_agent_graph import build_multi_agent_graph, load_mcp_tools

# Redis 连接：本地开发连 WSL2 里的 Redis（localhost 由 WSL2 自动转发）；
# Docker 部署时 compose 注入 REDIS_HOST=redis，指向 redis 容器，代码不用改
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
try:
    r.ping()
    print(f"[Redis] ✅ 已连接 {REDIS_HOST}:6379")
except redis.ConnectionError:
    print(f"[Redis] ⚠️ 连不上 Redis（{REDIS_HOST}:6379），请先在 WSL 里启动：sudo service redis-server start")

SESSIONS_DIR = Path("faiss_db/sessions")  # 会话索引都放这里（faiss_db 已在 .gitignore）
MAX_CACHED = 20                            # 内存最多缓存多少个重建好的会话
_cache: OrderedDict = OrderedDict()        # session_id -> 重建好的会话字典


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def _cache_put(session_id, session):
    _cache[session_id] = session
    _cache.move_to_end(session_id)
    while len(_cache) > MAX_CACHED:
        _cache.popitem(last=False)  # 淘汰最久没用的


def save_session(session_id, session, page_count):
    """上传建好索引后调用：重物落盘 + 元数据进 Redis + 缓存起来。
    session 形如 {vector_store, bm25, chunks, parent_docs, graph, llm}"""
    session_dir = SESSIONS_DIR / session_id
    (session_dir / "faiss_index").mkdir(parents=True, exist_ok=True)
    session["vector_store"].save_local(str(session_dir / "faiss_index"))
    with open(session_dir / "chunks.pkl", "wb") as f:
        pickle.dump(session["chunks"], f)
    with open(session_dir / "parent_docs.pkl", "wb") as f:
        pickle.dump(session["parent_docs"], f)
    with open(session_dir / "bm25.pkl", "wb") as f:
        pickle.dump(session["bm25"], f)

    r.hset(_key(session_id), mapping={
        "status": "ready",
        "created_at": datetime.now().isoformat(),
        "page_count": page_count,
        "child_count": len(session["chunks"]),
        "parent_count": len(session["parent_docs"]),
    })
    r.sadd("session:index", session_id)   # 用 Set 记全部会话，健康检查数用它
    _cache_put(session_id, session)


async def get_session(session_id, checkpointer=None, store=None):
    """对话时调用：内存缓存命中直接返回；未命中按 Redis 元数据从磁盘重建。
    返回 None 表示会话不存在（前端应提示先上传 PDF）。"""
    if not r.exists(_key(session_id)):
        return None
    if session_id in _cache:
        _cache.move_to_end(session_id)
        return _cache[session_id]
    return await _rebuild(session_id, checkpointer, store)


async def _rebuild(session_id, checkpointer, store):
    """从磁盘重建一个会话：FAISS + chunks + bm25 + 重编 graph。"""
    session_dir = SESSIONS_DIR / session_id
    index_dir = session_dir / "faiss_index"
    if not index_dir.exists():
        return None

    # 重启后第一次提问能看到这行，证明走的是"磁盘重建"路径而不是内存缓存
    print(f"[Session] 从磁盘重建会话 {session_id}")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
    with open(session_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(session_dir / "parent_docs.pkl", "rb") as f:
        parent_docs = pickle.load(f)
    try:
        with open(session_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
    except FileNotFoundError:
        bm25 = create_bm25(chunks)  # 兜底：pickle 丢了就从 chunks 重建

    # MCP 工具连不上就退回纯本地工具，不影响使用
    try:
        mcp_tools = await load_mcp_tools()
    except Exception:
        mcp_tools = []

    llm = get_llm()
    graph = build_multi_agent_graph(
        llm, vector_store, bm25, chunks, parent_docs,
        max_tool_rounds=5,
        checkpointer=checkpointer,
        store=store,
        extra_tools=mcp_tools,
    )
    session = {
        "vector_store": vector_store,
        "bm25": bm25,
        "chunks": chunks,
        "parent_docs": parent_docs,
        "graph": graph,
        "llm": llm,
    }
    _cache_put(session_id, session)
    return session


def delete_session(session_id):
    """删除会话：清 Redis 元数据 + 内存缓存（磁盘文件保留，需时再加清理）。"""
    r.delete(_key(session_id))
    r.srem("session:index", session_id)
    _cache.pop(session_id, None)


def active_session_count() -> int:
    return r.scard("session:index")
