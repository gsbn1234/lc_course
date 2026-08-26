"""检索层测试：不碰真实 embedding 模型和网络，用合成文档 + 假向量库。"""

import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from multi_agent.bm25 import create_bm25
from multi_agent.parent_retriever import parent_hybrid_retrieve
from multi_agent.retriever import bm25_search, vector_search


class FakeVectorStore:
    """模拟 FAISS 库：相似度搜索直接返回预设的 (doc, 距离) 列表，不加载 bge 模型。"""

    def __init__(self, docs_with_scores):
        self._docs = docs_with_scores

    def similarity_search_with_score(self, query, k=3):
        return self._docs[:k]


def _doc(content, parent_id):
    return Document(page_content=content, metadata={"parent_id": parent_id})


# 三组父子文档：child 是搜索引的小块，parent 是返回给用户的大块
CHILD_DOCS = [
    _doc("RAG是检索增强生成，先检索再生成", "p1"),
    _doc("LangGraph是多Agent协作框架", "p2"),
    _doc("FAISS是高效向量数据库", "p3"),
]
PARENT_DOCS = [
    _doc("RAG全称检索增强生成，先检索相关文档再生成回答，减少幻觉。", "p1"),
    _doc("LangGraph基于状态图编排多个智能体协作完成复杂任务。", "p2"),
    _doc("FAISS是Meta开源的向量数据库，支持海量向量近似检索。", "p3"),
]


def test_bm25_search_returns_most_relevant():
    bm25 = create_bm25(CHILD_DOCS)
    results = bm25_search("什么是RAG", bm25, CHILD_DOCS, k=2)
    assert results, "BM25 应至少返回一条结果"
    top_doc, top_score = results[0]
    assert "RAG" in top_doc.page_content
    assert 0 < top_score <= 1  # 归一化后的分数应在 (0, 1] 区间


def test_vector_search_converts_distance_to_similarity():
    store = FakeVectorStore([(_doc("RAG是检索增强生成", "p1"), 0.0)])  # 距离 0 = 完全相似
    results = vector_search("RAG", store, k=1)
    assert len(results) == 1
    _, sim = results[0]
    assert sim == 1.0  # 1/(1+0) = 1


def test_parent_hybrid_retrieve_maps_child_to_parent():
    bm25 = create_bm25(CHILD_DOCS)
    # 假向量库：让"RAG"命中的是 p1 的 child
    store = FakeVectorStore([
        (CHILD_DOCS[0], 0.5),   # p1 距离小 = 更相似
        (CHILD_DOCS[1], 1.5),   # p2 距离大 = 更不相似
    ])
    results = parent_hybrid_retrieve(
        "什么是RAG", store, bm25, CHILD_DOCS, PARENT_DOCS, top_k=3,
    )
    assert results
    top = results[0]
    # 返回的必须是 parent（内容更长），且属于 p1
    assert len(top["doc"].page_content) > len(CHILD_DOCS[0].page_content)
    assert top["doc"].metadata["parent_id"] == "p1"
    # 去重：同一 parent 不应重复出现
    contents = [r["doc"].page_content for r in results]
    assert len(contents) == len(set(contents))


class _FakeEmbeddings(Embeddings):
    """同一文本 → 同一向量（文本哈希当随机种子），不同文本向量不同。不加载 bge 模型。
    必须继承 langchain_core 的 Embeddings：新版 FAISS 靠 isinstance 判断，
    否则会把对象当普通函数直接调用，抛 TypeError。"""

    def _vec(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return rng.normal(size=16).astype("float32")

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_get_vector_store_force_rebuild(tmp_path, monkeypatch):
    """上传场景必须强制重建索引，不能复用全局 faiss_db/ 的旧索引（历史 bug）。"""
    from multi_agent import vector_db

    # 把全局索引路径指到临时目录，避免污染真实 faiss_db/
    monkeypatch.setattr(vector_db, "FAISS_PATH", str(tmp_path / "global_faiss"))

    docs_a = [_doc("苹果公司发布新款手机", "a1"), _doc("AI 大模型正在改变编程方式", "a2")]
    docs_b = [_doc("北京的秋天适合去香山看红叶", "b1"), _doc("长城是中国著名的景点", "b2")]
    emb = _FakeEmbeddings()

    # 1. 缓存模式第一次：建库并保存到全局路径
    vs1 = vector_db.get_vector_store(docs_a, emb)
    assert vs1.index.ntotal == len(docs_a)

    # 2. 缓存模式第二次传 docs_b：加载的是旧索引（docs_a）——脚本场景的"索引复用"是刻意的
    vs2 = vector_db.get_vector_store(docs_b, emb)
    hit = vs2.similarity_search("苹果", k=1)[0]
    assert hit.metadata["parent_id"] in {"a1", "a2"}, "缓存模式应复用旧索引，而不是重建"

    # 3. use_cache=False：必须强制基于 docs_b 重建，这才是上传场景该有的行为
    vs3 = vector_db.get_vector_store(docs_b, emb, use_cache=False)
    hit = vs3.similarity_search("香山", k=1)[0]
    assert hit.metadata["parent_id"] in {"b1", "b2"}, "强制重建后仍拿到旧文档"

    # 4. 强制重建不写全局缓存路径（vs1 建的文件仍在，证明没有污染）
    assert (tmp_path / "global_faiss").exists()
