import hashlib
import jieba

from .retriever import vector_search, bm25_search


def parent_hybrid_retrieve(
        question,
        child_vector_store,        # 小块 FAISS 向量库
        child_bm25,                # 小块 BM25 索引
        child_chunks,              # 小块 Document 列表
        parent_docs,               # 大块 Document 列表
        vector_weight=0.6,
        bm25_weight=0.4,
        top_k=10
):
    """
    Parent Document 混合检索:
      1. 在 child 粒度上做 FAISS + BM25 混合检索（小块搜得准）
      2. 通过 parent_id 映射回对应的 parent（大块上下文完整）
      3. 去重合并后返回 parent Document

    参数:
        question:             检索问题
        child_vector_store:   小块 FAISS 库
        child_bm25:           小块 BM25 索引
        child_chunks:         小块 Document 列表
        parent_docs:          大块 Document 列表（按 parent_id 索引）
        vector_weight:        FAISS 权重
        bm25_weight:          BM25 权重
        top_k:                最终返回的 parent 数量

    返回:
        大块结果列表 [{"doc": Document, "score": float}, ...]
    """
    # ===== 第一步：在 child 粒度上做混合检索 =====
    vector_docs = vector_search(question, child_vector_store, k=5)
    bm25_docs = bm25_search(question, child_bm25, child_chunks, k=5)

    merged = {}

    for doc, score in vector_docs:
        pid = doc.metadata.get("parent_id", "")
        key = pid
        merged[key] = {"doc": doc, "score": score * vector_weight, "parent_id": pid}

    for doc, score in bm25_docs:
        pid = doc.metadata.get("parent_id", "")
        key = pid
        weighted = score * bm25_weight
        if key in merged:
            merged[key]["score"] = max(merged[key]["score"], weighted)
        else:
            merged[key] = {"doc": doc, "score": weighted, "parent_id": pid}

    # ===== 第二步：child → parent 映射 =====
    # 构建 parent_id → Document 的索引
    parent_index = {
        p.metadata["parent_id"]: p for p in parent_docs
    }

    parent_results = {}
    for key, item in merged.items():
        pid = item["parent_id"]
        child_score = item["score"]

        if pid not in parent_index:
            continue

        parent_doc = parent_index[pid]

        # 用 parent 内容做键（字符串，跨进程稳定）
        content_key = hashlib.md5(parent_doc.page_content.encode()).hexdigest()

        if content_key in parent_results:
            # 同一 parent 被多个 child 命中，取最高分
            parent_results[content_key]["score"] = max(
                parent_results[content_key]["score"], child_score   #把更大的浮点分数重新赋值给内层字典的score键（赋值操作）
            )
        else:
            parent_results[content_key] = {
                "doc": parent_doc,
                "score": child_score
            }

    final_results = sorted(
        parent_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    print(f"[Parent] {len(parent_results)} parents → top{len(final_results)}")
    return final_results