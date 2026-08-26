# 前端镜像：Streamlit，只需 streamlit + requests，比后端镜像小得多
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    streamlit==1.61.1 requests==2.34.2

COPY streamlit_1/app.py ./streamlit_1/app.py

EXPOSE 8501

# --server.address 0.0.0.0：Streamlit 默认绑 127.0.0.1，容器里外部访问不到
CMD ["streamlit", "run", "streamlit_1/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
