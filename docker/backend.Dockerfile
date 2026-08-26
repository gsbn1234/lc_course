# 后端镜像：FastAPI + LangGraph Agent-RAG
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装 torch。本机是 CPU 版（2.13.0+cpu），但 CPU 专用 wheel 国内无镜像
# （官方源被 GFW 限速到 60KB/s）。清华 PyPI 的 torch 2.13.0（502MB，打包 CUDA 运行时）
# 无 GPU 时自动跑 CPU，功能完全等价，清华源 3.6MB/s 约 2 分钟下完。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.13.0

# 其余依赖全走清华源；torch 已装好，pip 这里直接跳过不会重装
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 业务代码。.env 故意不 COPY —— 里面是 API Key，只允许运行期由 compose env_file 注入
COPY multi_agent/ ./multi_agent/
COPY streamlit_1/ ./streamlit_1/
COPY mcp/ ./mcp/
COPY main.py ./main.py

EXPOSE 8000

# 0.0.0.0：容器端口要对宿主机/别的容器可见，不能绑 127.0.0.1
CMD ["uvicorn", "streamlit_1.backend:app", "--host", "0.0.0.0", "--port", "8000"]
