"""对话接口 user_id 的透传、校验与记忆隔离测试。

背景：user_id 是长期记忆的 Store 命名空间键（("users", user_id)），
后端原本把它写死成 "21702"，等于所有用户共用一份偏好记忆——
A 说"记住我喜欢简洁回答"，B 也读得到。

这里锁死四件事：
  1. 请求里的 user_id 原样进图 config（不再写死）
  2. 缺 user_id → 422：没有默认值，免得"忘了传"又悄悄退回共用命名空间
  3. 首尾空格被剥掉（否则 "alice" 和 "alice " 会变成两个互不相通的用户）
  4. 真实节点函数确实按 user_id 隔离读写

不碰网络：rewrite_query 换成桩，LLM 用假流式对象。
"""
import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, SystemMessage
from langgraph.store.memory import InMemoryStore
from starlette.testclient import TestClient

import streamlit_1.backend as backend
from multi_agent import multi_agent_graph as mag


# ========== 一、HTTP 层：透传与校验 ==========

class _FakeGraph:
    """只记录拿到的 config，然后回一个终止态状态块。"""

    def __init__(self):
        self.config = None

    async def astream(self, state, config=None, stream_mode=None):
        self.config = config
        yield ("values", {"next_agent": "finish"})


def _post_chat(payload):
    graph = _FakeGraph()

    async def fake_get_session(session_id, checkpointer=None, store=None):
        return {"graph": graph}

    with patch.object(backend, "get_session", fake_get_session):
        client = TestClient(backend.app)
        resp = client.post("/api/chat-stream", json=payload)
    return resp, graph


def test_user_id_is_passed_through_to_graph():
    resp, graph = _post_chat(
        {"session_id": "s1", "question": "你好", "user_id": "alice"}
    )
    assert resp.status_code == 200, resp.text
    assert graph.config["configurable"]["user_id"] == "alice", (
        "user_id 没透传到图——又被写死了"
    )
    assert graph.config["configurable"]["thread_id"] == "s1"


def test_user_id_is_stripped():
    """首尾空格剥掉，避免 "alice" 和 "alice " 变成两个用户。"""
    resp, graph = _post_chat(
        {"session_id": "s1", "question": "你好", "user_id": "  alice  "}
    )
    assert resp.status_code == 200, resp.text
    assert graph.config["configurable"]["user_id"] == "alice"


def test_missing_user_id_is_rejected():
    """字段没有默认值：漏传必须报错，不能默默退回共用命名空间。"""
    resp, _ = _post_chat({"session_id": "s1", "question": "你好"})
    assert resp.status_code == 422, resp.text


def test_blank_or_overlong_user_id_is_rejected():
    for bad in ("", "   ", "x" * 65):
        resp, _ = _post_chat(
            {"session_id": "s1", "question": "你好", "user_id": bad}
        )
        assert resp.status_code == 422, f"user_id={bad[:10]!r} 应被拒"


# ========== 二、图节点：真实读写是否按 user_id 隔离 ==========

def test_memory_write_is_namespaced_by_user_id():
    """A 说"记住xxx"只写进 A 的命名空间，B 查不到。"""
    store = InMemoryStore()
    with patch.object(mag, "rewrite_query", lambda q, llm: q):
        mag.rewrite_query_node(
            {"question": "记住我喜欢简洁回答"},
            None,                                   # llm 被 rewrite_query 桩挡掉了
            {"configurable": {"user_id": "alice"}},
            store,
        )

    saved = store.get(("users", "alice"), "style")
    assert saved is not None, "偏好没写进 alice 的命名空间"
    assert saved.value["data"] == "我喜欢简洁回答"
    assert store.get(("users", "bob"), "style") is None, "偏好泄漏到了 bob 的命名空间"


class _FakeStreamLLM:
    """假流式 LLM：记录收到的 messages，回一个分片。"""

    def __init__(self):
        self.seen_messages = None

    async def astream(self, messages, config=None):
        self.seen_messages = messages
        yield AIMessageChunk(content="好的")


async def _writer_sees(store, user_id):
    llm = _FakeStreamLLM()
    await mag.writer_agent_node(
        {"writer_messages": []}, llm, {"configurable": {"user_id": user_id}}, store
    )
    return llm.seen_messages


def test_memory_read_is_namespaced_by_user_id():
    """alice 的偏好会被注入成 SystemMessage；bob 读不到，所以没有注入。"""
    store = InMemoryStore()
    store.put(("users", "alice"), "style", {"data": "回答要简洁"})

    alice_msgs = asyncio.run(_writer_sees(store, "alice"))
    bob_msgs = asyncio.run(_writer_sees(store, "bob"))

    assert alice_msgs, "alice 的 writer 没拿到任何消息"
    assert isinstance(alice_msgs[0], SystemMessage), "alice 的偏好没被注入"
    assert "回答要简洁" in alice_msgs[0].content

    assert not any(isinstance(m, SystemMessage) for m in bob_msgs), (
        "bob 读到了 alice 的偏好——记忆串了"
    )
