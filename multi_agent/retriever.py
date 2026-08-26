"""
检索基础函数。

只保留当前系统仍在使用的最小集合：
- vector_search  ：FAISS 向量相似度检索（parent_retriever.py 依赖）
- bm25_search    ：BM25 关键词检索（parent_retriever.py 依赖）

已废弃删除：
- hybrid_retrieve / multi_hybrid_retrieve：
  旧版单粒度混合检索，已被 parent_retriever.py 的
  parent_hybrid_retrieve（child 小粒度检索 + parent 大粒度返回）取代。
  且旧实现用内置 hash() 做去重键，跨进程不稳定。
  如需找回，见备份 zip：lc_course_已删除代码备份_20260822.zip
"""

import jieba



def vector_search(
        question,
        vector_store,
        k=3
):

    results = vector_store.similarity_search_with_score(

        question,

        k=k

    )


    docs=[]


    for doc,score in results:

        similarity=1/(1+score)

        docs.append(
            (
                doc,
                similarity
            )
        )


    return docs




def bm25_search(
        question,
        bm25,
        chunks,
        k=3
):


    # ★ 必须 .lower()：create_bm25 建索引时对语料做了小写化（text.lower()），
    # 查询侧如果不转小写，英文词（RAG/LangGraph/API...）就会大小写对不上被静默丢弃，
    # 检索退化成"谁短谁分高"。由 pytest 用例 test_bm25_search_returns_most_relevant 捕获。
    query_tokens = jieba.lcut(
        question.lower()
    )


    scores=bm25.get_scores(
        query_tokens
    )


    ranked=sorted(

        enumerate(scores),

        key=lambda x:x[1],

        reverse=True

    )[:k]


    results=[]


    max_score = ranked[0][1] if ranked and ranked[0][1]>0 else 1


    for idx,score in ranked:

        normalized=score/max_score


        results.append(

            (
                chunks[idx],
                normalized
            )

        )


    return results
