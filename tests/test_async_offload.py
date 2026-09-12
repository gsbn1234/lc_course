"""异步不阻塞测试：证明同步重活已经离开事件循环。

背景：upload_pdf / _rebuild 原本把 PDF 解析、FAISS 建库、jieba 分词这些
同步阻塞操作直接写在 async 函数里，会把事件循环占死——上传期间其它请求
（包括 /api/health）全部排队。改成 asyncio.to_thread 后，这些活在线程池线程里跑。

测试全部用桩替换，不碰真实 embedding / FAISS / Redis / MCP。
"""
import asyncio
import threading
import time
from unittest.mock import patch

from langchain_core.documents import Document
from starlette.testclient import TestClient

import streamlit_1.backend as backend
import streamlit_1.session_store as session_store


def _tname() -> str:
    return threading.current_thread().name


# ========== 一、_rebuild：真实代码路径 ==========

def test_rebuild_does_not_block_event_loop():
    """_rebuild 的阻塞部分在线程池跑：期间心跳协程照常跳。"""
    ticks = 0

    def slow_sync(session_id):
        time.sleep(0.5)      # 模拟 FAISS 读盘 + jieba 分词
        return None          # 返回 None → _rebuild 提前返回，不需要 Redis

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    async def main():
        hb = asyncio.create_task(heartbeat())
        with patch.object(session_store, "_rebuild_sync", slow_sync):
            result = await session_store._rebuild("sid", None, None)
        hb.cancel()
        return result

    assert asyncio.run(main()) is None
    # 0.5s / 0.05s ≈ 10 次；给足余量只要求 ≥5
    assert ticks >= 5, f"事件循环被阻塞：0.5s 内心跳只跳了 {ticks} 次"


def test_control_sync_call_in_coroutine_does_block():
    """对照实验：证明上面的心跳断言真的能发现阻塞，不是空断言。

    把同样的 sleep 直接写在协程里（= 改造前的写法），心跳应该几乎不跳。
    """
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    async def main():
        hb = asyncio.create_task(heartbeat())
        time.sleep(0.5)      # 旧写法：同步阻塞占死事件循环
        hb.cancel()

    asyncio.run(main())
    assert ticks <= 1, f"对照组本该被阻塞却跳了 {ticks} 次——说明心跳检测无效"


def test_rebuild_sync_runs_in_worker_thread():
    """_rebuild 里被调用的 _rebuild_sync 必须不在事件循环线程上跑。"""
    seen = {}

    def fake_sync(session_id):
        seen["thread"] = _tname()
        return None

    async def main():
        with patch.object(session_store, "_rebuild_sync", fake_sync):
            await session_store._rebuild("sid", None, None)

    asyncio.run(main())

    assert seen, "_rebuild_sync 没被调用"
    assert seen["thread"] != _tname(), (
        f"_rebuild_sync 跑在主线程 {seen['thread']} 上——没走线程池"
    )


# ========== 二、upload_pdf：整条 HTTP 链路 ==========

def test_upload_pdf_heavy_work_runs_off_event_loop():
    """上传接口：重活在线程池，轻活（图编译）留在事件循环——分工精确。"""
    thread_of = {}          # 被桩函数名 -> 执行它的线程名
    child = [Document(page_content="child", metadata={})]
    parent = [Document(page_content="parent", metadata={})]

    def stub(tag, result):
        def fn(*args, **kwargs):
            thread_of[tag] = _tname()
            return result
        return fn

    class FakeLoader:
        def __init__(self, path):
            pass

        def load(self):
            thread_of["PyPDFLoader.load"] = _tname()
            return [Document(page_content="page", metadata={})]

    async def fake_load_mcp_tools(domain):
        # 这个 await 一定在事件循环上执行，用它记录"事件循环线程"是谁
        thread_of["load_mcp_tools"] = _tname()
        return []

    with patch.object(backend, "PyPDFLoader", FakeLoader), \
            patch.object(backend, "split_parent_child", stub("split_parent_child", (child, parent))), \
            patch.object(backend, "get_embeddings", stub("get_embeddings", object())), \
            patch.object(backend, "get_vector_store", stub("get_vector_store", object())), \
            patch.object(backend, "create_bm25", stub("create_bm25", object())), \
            patch.object(backend, "get_llm", stub("get_llm", object())), \
            patch.object(backend, "build_multi_agent_graph", stub("build_multi_agent_graph", object())), \
            patch.object(backend, "save_session", stub("save_session", None)), \
            patch.object(backend, "load_mcp_tools", fake_load_mcp_tools):
        client = TestClient(backend.app)
        resp = client.post(
            "/api/upload-pdf",
            files=[("files", ("a.pdf", b"%PDF-1.4 fake content", "application/pdf"))],
            data={"domain": "general"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["child_chunks"] == 1
    assert body["parent_chunks"] == 1

    assert "load_mcp_tools" in thread_of, "接口没跑到 MCP 加载那步"
    loop_thread = thread_of["load_mcp_tools"]

    # 重活：必须离开事件循环线程
    for tag in ("PyPDFLoader.load", "split_parent_child", "get_vector_store",
                "create_bm25", "save_session"):
        assert tag in thread_of, f"{tag} 没被调用"
        assert thread_of[tag] != loop_thread, (
            f"{tag} 仍跑在事件循环线程 {loop_thread} 上——阻塞没修掉"
        )

    # 轻活：故意留在事件循环上（只建对象/编译图，不做 IO）
    for tag in ("get_llm", "build_multi_agent_graph"):
        assert thread_of.get(tag) == loop_thread, (
            f"{tag} 不该被丢进线程池（无 IO，白付线程切换开销）"
        )
