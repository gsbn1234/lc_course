"""接口鉴权 + 健康检查的依赖探测测试。

背景（两件事）：
  1. 没有鉴权。CORS 只锁来源，它约束的是浏览器；curl / requests 直接打 :8000
     完全绕开，谁都能上传 PDF、跑 Agent、烧 token。加一层 API Key。
  2. /api/health 不查依赖。原来固定返回 {"status": "ok"}，Redis 挂了照样说 ok，
     探活方被蒙在鼓里。合格的健康检查要逐个报告组件状态。

这里锁死的行为：
  · 没配 Key → 放行（本地开发默认）；配了 Key → 不带/带错都是 401
  · Bearer 和 X-API-Key 两种头都认
  · /api/health 不需要鉴权（探针没凭据，要鉴权的话探针永远 401）
  · Redis / 索引目录挂了 → 503；只有 MCP 出问题 → degraded 但仍是 200

不碰网络、不碰真 Redis：外部依赖全换成桩。
"""
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

import streamlit_1.backend as backend


# ========== 一、鉴权 ==========

class _FakeGraph:
    """只回一个终止态状态块，让请求能走到流式响应结束。"""

    async def astream(self, state, config=None, stream_mode=None):
        yield ("values", {"next_agent": "finish"})


async def _fake_get_session(session_id, checkpointer=None, store=None):
    return {"graph": _FakeGraph()}


def _chat(headers):
    """打一次对话接口，返回响应。用桩会话，绕开真实的索引重建。"""
    with patch.object(backend, "get_session", _fake_get_session):
        return TestClient(backend.app).post(
            "/api/chat-stream",
            json={"session_id": "s1", "question": "你好", "user_id": "alice"},
            headers=headers,
        )


def test_no_key_configured_allows_anonymous(monkeypatch):
    """没配 BACKEND_API_KEY = 本地开发模式，不校验，请求照常放行。"""
    monkeypatch.setattr(backend, "API_KEY", "")
    assert _chat({}).status_code == 200


def test_key_configured_rejects_missing_header(monkeypatch):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    resp = _chat({})
    assert resp.status_code == 401, resp.text
    # 带上 WWW-Authenticate，符合 HTTP 认证失败的约定
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    assert "API Key" in resp.json()["detail"]


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer wrong-key"},          # 值不对
        {"Authorization": "Bearer s3cret-extra"},       # 只是前缀对，也要拒
        {"Authorization": "Bearer "},                   # 空 token
        {"Authorization": "s3cret"},                    # 少了 Bearer 前缀 → 不是可识别的凭据
        {"Authorization": "Basic czNjcmV0"},            # 换了认证方案
        {"X-API-Key": "wrong-key"},
        {"X-API-Key": ""},
    ],
)
def test_key_configured_rejects_bad_credentials(monkeypatch, headers):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    resp = _chat(headers)
    assert resp.status_code == 401, f"{headers} 应被拒，实际 {resp.status_code}"


def test_bearer_header_accepted(monkeypatch):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    assert _chat({"Authorization": "Bearer s3cret"}).status_code == 200


def test_x_api_key_header_accepted(monkeypatch):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    assert _chat({"X-API-Key": "s3cret"}).status_code == 200


def test_bearer_prefix_is_case_insensitive(monkeypatch):
    """HTTP 认证方案名大小写不敏感（RFC 7235），别只认小写。"""
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    assert _chat({"Authorization": "bearer s3cret"}).status_code == 200


def test_upload_route_is_also_protected(monkeypatch):
    """上传接口同样挂了依赖：不带 Key 时在校验文件之前就被挡下（401 而不是 422）。"""
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    resp = TestClient(backend.app).post("/api/upload-pdf")
    assert resp.status_code == 401, resp.text


# ========== 二、健康检查 ==========

def _health(redis=None, mcp=None, sessions_dir=None, api_key="s3cret"):
    """打一次健康检查，三个依赖各换成指定桩。"""
    patches = [
        patch.object(backend, "API_KEY", api_key),
        patch.object(backend, "redis_status", lambda: redis),
        patch.object(backend, "mcp_status", lambda: mcp),
    ]
    if sessions_dir is not None:
        patches.append(patch.object(backend, "SESSIONS_DIR", sessions_dir))
    for p in patches:
        p.start()
    try:
        return TestClient(backend.app).get("/api/health")
    finally:
        for p in patches:
            p.stop()


def test_health_needs_no_auth(monkeypatch, tmp_path):
    """配了 Key 也不拦健康检查——探针手里没有凭据。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 3},
        mcp={"status": "not_started"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200, resp.text


def test_health_ok_when_all_dependencies_up(tmp_path):
    resp = _health(
        redis={"status": "ok", "active_sessions": 3},
        mcp={"status": "connected"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["active_sessions"] == 3
    assert set(body["checks"]) == {"redis", "session_dir", "mcp", "checkpointer"}
    assert body["checks"]["session_dir"]["path"] == str(tmp_path)


def test_health_503_when_redis_down(tmp_path):
    """Redis 是硬依赖（会话元数据全在里面）→ 503，让编排系统摘流量。"""
    resp = _health(
        redis={"status": "error", "detail": "ConnectionError: refused"},
        mcp={"status": "connected"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["status"] == "error"
    assert "refused" in body["checks"]["redis"]["detail"]
    assert body["active_sessions"] is None      # 探不到就没有这个数，不编一个


def test_health_503_when_session_dir_missing(tmp_path):
    """索引目录不见了，上传/重建全会失败，同样算硬依赖故障。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 0},
        mcp={"status": "connected"},
        sessions_dir=tmp_path / "不存在的目录",
    )
    assert resp.status_code == 503
    assert resp.json()["status"] == "error"


def test_health_degraded_but_200_when_mcp_broken(tmp_path):
    """MCP 只是工具降级（连不上会退回纯本地检索），不该把整个服务判死。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 1},
        mcp={"status": "error", "detail": "session 为空"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_health_mcp_not_started_is_not_degraded(tmp_path):
    """MCP 是懒加载：没人传过该领域的 PDF 就还没拉起来，这是正常初始态。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 0},
        mcp={"status": "not_started"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_redis_probe_failure_is_reported_not_raised():
    """探测函数不抛异常：健康检查的职责是"报告是谁挂了"，
    抛出去调用方只会拿到 500，看不出是哪个组件。"""
    with patch("streamlit_1.session_store.r") as fake_r:
        fake_r.ping.side_effect = ConnectionError("refused")
        status = backend.redis_status()
    assert status["status"] == "error"
    assert "ConnectionError" in status["detail"]
