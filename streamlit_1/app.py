"""
Streamlit 前端 — 前后端分离版。

演进变化：
  原版：Streamlit 直接调 build_multi_agent_graph → graph.stream()
  现在：Streamlit 调 FastAPI → API 跑 Agent → SSE 流式返回 → 前端解析渲染

架构：Streamlit(8501) ↔ HTTP/SSE ↔ FastAPI(8000) ↔ LangGraph Agent
"""

import os
import requests #Python 最常用 HTTP 请求库，用来调用 FastAPI 接口，上传文件、发起对话、接收流式返回；
import json
import uuid #生成长期记忆的身份标识（user_id）
import streamlit as st #`streamlit as st`：网页 UI 框架。

# 本地开发默认 127.0.0.1:8000；Docker 里 compose 注入 BACKEND_URL=http://backend:8000
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# 后端开了鉴权（BACKEND_API_KEY）时，前端得带上同一个 Key，否则会被 401 挡在门外。
# 两个服务读的是同一个变量，compose 里用 env_file / environment 各注入一次。
API_KEY = os.getenv("BACKEND_API_KEY", "").strip()
# 鉴权头：没配 Key 就不发这个头（后端那边同步也是"没配就不校验"，两边行为一致）
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

# 领域代码 → 中文名（下拉框和上传成功提示共用这一份，只改这里）
# 注意：后端 mcp_tools/registry.py 里也有一份 DOMAIN_LABELS（它给 registry 用），
# 两侧各自维护，增领域时记得两边同步。
DOMAIN_MAP = {
    "ai_learning": "AI 学习",
    "general": "通用",
    "finance": "金融",
    "medical": "医疗",
}

st.set_page_config(page_title="Adaptive Research Agent")

# ========== 1. 初始化 session_state ==========
if "initialized" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.messages = []
    # 长期记忆的身份：每个浏览器会话生成一个，随对话请求发给后端。
    # 后端拿它当 Store 命名空间，不同身份的记忆互不可见——这样"记住xxx"
    # 不会串到别人身上。放 session_state 里 → 同一次浏览器会话内保持不变
    # （刷新页面会重新生成、换一份新记忆；想跨刷新沿用同一份，就把它换成
    # 侧边栏里让用户自己填的名字）。
    st.session_state.user_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.initialized = True

# ========== 2. 侧边栏：上传 PDF + 参数配置 ==========
with st.sidebar:
    st.header("📄 知识库")
    #技术点：st.file_uploader — Streamlit 把用户上传的文件包装成 UploadedFile 对象，存在内存里，此时文件还没离开浏览器所在机器
    uploaded_files = st.file_uploader(
        "上传 PDF 文件",
        type="pdf",
        accept_multiple_files=True,
        help="支持同时上传多个 PDF，上传后点击「构建索引」",
    )

    # 知识库领域：决定 Researcher 挂哪些 MCP 工具（见 mcp_tools/registry.py）
    domain = st.selectbox(
        "知识库领域",
        options=["ai_learning", "general", "finance", "medical"],
        format_func=lambda d: DOMAIN_MAP[d],
        help="选择上传 PDF 所属领域。AI 学习领域会为 Researcher 挂上 "
             "GitHub / arXiv / HuggingFace 三个真实检索工具",
    )

    st.header("⚙️ 参数")
    max_rounds = st.slider("最大搜索轮数", 1, 10, 5)

    if st.button("🔄 构建/重建索引", use_container_width=True):#按钮组件；`use_container_width=True` = 按钮宽度铺满侧边栏。
        if not uploaded_files:
            st.warning("请先上传 PDF 文件")
        else:
            with st.spinner("正在上传 PDF 并构建索引..."):
                # 构建上传请求 → 发到 FastAPI
#                 """
#                  requests 上传文件标准格式
# 上传文件时 post 请求需要构造 files 参数：
# `(表单字段名, (文件名, 文件二进制, MIME类型))`
# - 表单 key 统一叫 `files`，后端 FastAPI 接收`List[UploadFile]`
# - `f.getvalue()` 读取 pdf 二进制字节，不会写入本地磁盘，直接内存上传后端。
#                 """
                files = []
                for f in uploaded_files:
                    files.append(
                        ("files", (f.name, f.getvalue(), "application/pdf"))
                    )

                try:
                    resp = requests.post(
                        f"{BACKEND}/api/upload-pdf",
                        files=files,
                        data={"domain": domain},   # 领域：决定挂哪些 MCP 工具
                        headers=AUTH_HEADERS,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.messages = []  # 新知识库，清空历史对话
                        raw_domain = data.get("domain", "")
                        domain_name = DOMAIN_MAP.get(raw_domain, raw_domain)
                        st.success(
                            f"索引构建完成！{data['child_chunks']} 个 child chunk，"
                            f"{data['parent_chunks']} 个 parent chunk"
                            f"（领域：{domain_name}）"
                        )
                    else:
                        st.error(f"后端错误：{resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("无法连接后端，请先启动 FastAPI：uvicorn streamlit_1.backend:app --port 8000 --reload")

    # 显示当前状态
    if st.session_state.session_id:
        st.info(f"✅ 索引已就绪（会话: {st.session_state.session_id}）")
    else:
        st.warning("⚠️ 请上传 PDF 并构建索引")

    # 把记忆身份显示出来，便于验证隔离效果：
    # 换个浏览器（或刷新页面）就是一个新身份，之前"记住"的偏好读不到了。
    st.caption(f"🧠 记忆身份：{st.session_state.user_id}")

# ========== 3. 主区域 ==========
st.title("Adaptive Research Agent")
st.caption("Multi-Agent 协作检索系统 — Supervisor + Researcher + Writer")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ========== 4. 用户输入 ==========
prompt = st.chat_input("输入你的问题，Agent 会自动搜索本地库和互联网...")

if prompt:
    if not st.session_state.session_id:
        st.warning("请先在侧边栏上传 PDF 并点击「构建索引」")
    else:
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 调用 FastAPI 流式对话
        with st.chat_message("assistant"):
            status_container = st.empty()
            answer_container = st.empty()

            tool_logs = []
            answer = ""

            try:
                resp = requests.post(
                    f"{BACKEND}/api/chat-stream",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": prompt,
                        "max_tool_rounds": max_rounds,
                        "user_id": st.session_state.user_id,   # 长期记忆的身份
                    },
                    headers=AUTH_HEADERS,
                    stream=True,
                    timeout=120,
                )

                if resp.status_code == 200:
                    for line in resp.iter_lines(): #`resp.iter_lines()` 逐行读取 SSE 字节流。
                        if not line:
                            continue  #跳过数据流里面的空行，SSE 协议经常有空行作为分隔。
                        line = line.decode() #bytes 字节 → utf‑8 字符串。
                        if not line.startswith("data: "):
                            continue #SSE 协议标准格式：后端每一条推送消息固定前缀 `data:`不是该格式的行全部跳过。
                        data = json.loads(line[6:]) #切掉前缀`data: `（6 个字符），剩下字符串解析成 json 字典。

                        event_type = data.get("type")
                        if event_type == "status":
                            agent = data.get("agent")
                            if agent == "researcher":
                                status_container.info(
                                    f"🔍 Researcher 第 {data.get('rounds', '?')} 轮搜索中..."
                                )
                        elif event_type == "writer_start":
                            # 新一轮写作开始：清掉上一轮草稿（Reviewer 打回重写时）
                            status_container.info("✍️ Writer 正在撰写答案...")
                            answer = ""
                            answer_container.markdown("")
                        elif event_type == "token":
                            # 逐字追加，实现打字机效果
                            answer += data["content"]
                            answer_container.markdown(answer)
                        elif event_type == "tool_call":
                            tool_logs.append(
                                f"🔧 {data['name']}({data['args']})"
                            )
                        elif event_type == "done":
                            status_container.empty()
                            # 终稿（含 Reviewer 修订），覆盖流式内容，保证显示最终版本
                            answer = data["answer"]
                            answer_container.markdown(answer)
                        elif event_type == "error":
                            status_container.empty()
                            st.error(f"后端处理出错：{data.get('message', '未知错误')}")
                else:
                    st.error(f"后端错误：{resp.text}")
            except requests.exceptions.ChunkedEncodingError:
                st.error("后端响应中途中断，请查看 uvicorn 终端的报错")
            except requests.exceptions.ConnectionError:
                st.error("无法连接后端，请先启动 FastAPI")

        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})

        if tool_logs:
            with st.expander("🔍 查看检索过程"):
                for log in tool_logs:
                    st.text(log)
