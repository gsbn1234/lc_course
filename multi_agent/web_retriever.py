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
    tool = TavilySearchResults(max_results=k)    #工具本身能力：联网搜索，返回每条网页的正文摘要、url、标题等原始字典信息。
    raw_results = tool.invoke({"query": question})  #raw_results：Tavily 原始返回值，列表。列表内每个元素是原生字典，结构固定
                        #[  {"content": "网页摘要文本", "url": "网页链接", "title": "网页标题", ...} , ... ]。此时是原生字典，没有 Document、没有 score，必须手动包装。
    docs = []
    for r in raw_results:
        # Tavily 返回 {"content": "...", "url": "...", ...}
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