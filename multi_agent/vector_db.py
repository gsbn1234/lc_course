import os

from langchain_community.vectorstores import FAISS

from .config import FAISS_PATH



def get_vector_store(chunks, embeddings, use_cache=True):
    """
    获取向量库。

    use_cache=True（默认）：faiss_db/ 存在就加载复用，否则新建并保存。
                          适合 main.py / 评估脚本这类"索引建一次、跑多次"的场景。

    use_cache=False：强制基于 chunks 重建，不读全局缓存也不写缓存。
                    适合后端上传——每个会话必须用刚上传的文档建索引，
                    不能被全局 faiss_db/ 里的旧索引污染（历史 bug）。
    """
    if use_cache and os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vector_store = FAISS.from_documents(chunks, embeddings)
    if use_cache:
        vector_store.save_local(FAISS_PATH)
    return vector_store