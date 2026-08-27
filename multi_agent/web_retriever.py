"""
Tavily 网页搜索检索器。
返回和 parent_hybrid_retrieve 相同的数据结构，保证后续节点可以无差别处理。
"""
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults


def web_search(question, k=5):
    """
    用 Tavily 搜索互联网，返回和本地检索统一格式的结果。

    参数:
        question: 搜索问题
        k:        返回结果数量

    返回:
        [{"doc": Document, "score": float}, ...]
        和 parent_hybrid_retrieve() 返回值结构完全一致
    """
    tool = TavilySearchResults(max_results=k)
    raw_results = tool.invoke({"query": question})

    # 兼容 Tavily 的几种返回形态：
    #   list[dict]         正常（每条 {"content","url","title","score"}）
    #   dict               某些版本返回 {"results": [...]}
    #   str                Tavily 出错时 _run 会把异常 repr 成字符串返回
    #                      （见 langchain_community 的 TavilySearchResults._run），
    #                      也偶有 JSON 文本。此时联网失败，返回空让上游走本地检索兜底。
    if isinstance(raw_results, str):
        print(f"\n[Web Search] Tavily 返回异常（{raw_results[:100]}），本次联网检索降级为空")
        return []
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", [])

    docs = []
    for r in raw_results:
        # 极端兜底：个别条目不是 dict 也跳过，绝不崩在 .get 上
        if not isinstance(r, dict):
            continue
        content = r.get("content", "")
        url = r.get("url", "")

        doc = Document(
            page_content=content,
            metadata={"source": url, "type": "web"}
        )
        docs.append({
            "doc": doc,
            "score": 1.0      # 网页结果没有向量分数，统一给 1.0
        })

    print(f"\n[Web Search] Tavily 返回 {len(docs)} 条结果")
    return docs[:k]
