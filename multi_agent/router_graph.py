"""
基于 LangGraph 的 Query Routing RAG 工作流。

图结构:

    [START]
      │
      ▼
   rewrite_query_node      ← LLM 改写问题，存入 state["rewrite_question"]
      │
      ▼
   classify_route_node     ← LLM 判断走 "local" 还是 "web"
      │
      ├─ local ──→ local_retrieve_node   ← parent_hybrid_retrieve
      │
      └─ web ────→ web_retrieve_node     ← Tavily 网页搜索
      │                     │
      └─────────┬───────────┘
                ▼
      rerank_compress_node  ← CrossEncoder 精排 + 上下文压缩
                │
                ▼
         generate_node      ← DeepSeek 生成最终回答
                │
                ▼
              [END]
"""
import logging
from typing import TypedDict, List   #自定义 LangGraph 全局状态的类型约束，规范状态字段类型；

from langgraph.graph import StateGraph, START, END  #LangGraph 核心，构建状态图；START 起点、END 终点；
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .query_rewrite import rewrite_query
from .multi_query import generate_queries
from .parent_retriever import parent_hybrid_retrieve
from .web_retriever import web_search
from .reranker import rerank, get_reranker
from .context_compressor import compress_documents
from .prompt import prompt as answer_prompt
from .hyde import generate_hypothetical_answer

logger = logging.getLogger(__name__)


# ========== 1. 定义状态 ==========

class RAGState(TypedDict):
    """
    图中流转的状态字典。每个节点读取自己需要的字段，写入自己产出的字段。
    """
    question: str               # 用户原始问题（输入）
    rewrite_question: str       # LLM 改写后的检索问题
    hyde_answer: str            #假想答案
    route: str                  # "local" 或 "web"
    docs: List[dict]            # 检索结果 [{"doc": Document, "score": float}, ...]
    final_docs: List[dict]      # 精排 + 压缩后的最终文档
    answer: str                 # LLM 最终回答（输出）


# ========== 2. 分类 prompt ==========

router_prompt = ChatPromptTemplate.from_template("""
你是一个查询路由器。

判断用户问题应该从哪里获取信息：

规则：
- 如果问题是关于本地知识库内容（技术文档、概念解释、原理说明），输出 local
- 如果问题需要实时信息（新闻、最新动态、实时数据、天气、股价），输出 web
- 如果问题包含"最新"、"今天"、"现在"、"新闻"等词，输出 web
- 如果问题点名了具体模型/产品/框架的版本或代际（如 R1、V3、GPT-4o、iOS 18），
  问的是它们之间的对比、改进或新特性，输出 web —— 这类信息迭代快，本地知识库收录不了

只输出一个词：local 或 web，不要输出其他内容。

用户问题：
{question}

路由结果：""")


def classify_route(question, llm):
    """让 LLM 判断问题走本地检索还是网页搜索。"""
    chain = router_prompt | llm
    response = chain.invoke({"question": question})
    route = response.content.strip().lower()
    if route not in ("local", "web"):
        route = "local"   # 兜底：无法判断时走本地
    return route


# ========== 3. 定义各节点函数 ==========

def rewrite_query_node(state: RAGState, llm):
    """
    节点 1：改写问题。
    输入 state["question"]，输出 state["rewrite_question"]。
    """
    rewrite_question = rewrite_query(state["question"], llm)
    return {"rewrite_question": rewrite_question}


def hyde_node(state: RAGState, llm):
    """
    节点 1.5：生成假想答案。
    输入 state["rewrite_question"]，输出 state["hyde_answer"]。

    用 LLM 编一段假答案，后续检索用这段答案当查询词，
    让 Embedding 落在"文档空间"而非"问题空间"。
    """
    hyde_answer = generate_hypothetical_answer(state["rewrite_question"], llm)
    return {"hyde_answer": hyde_answer}



def classify_route_node(state: RAGState, llm):
    """
    节点 2：路由分类。
    输入 state["question"]，输出 state["route"]。
    """
    route = classify_route(state["question"], llm)
    logger.info("[Router] 问题分类: %s", route)
    return {"route": route}


def local_retrieve_node(state: RAGState, vector_store, bm25, chunks, parent_docs, llm,use_hyde=False):
    """
    节点 3a：本地知识库检索。
    1. 生成多角度查询问题
    2. 对每个问题调 parent_hybrid_retrieve（小块搜、大块取）
    3. 跨查询结果合并去重
    """
    # 基于改写后的问题生成多角度查询（一次 LLM 调用）
    query_source = state.get("hyde_answer", state["rewrite_question"]) if use_hyde else state["rewrite_question"]
    queries = generate_queries(query_source, llm, num_queries=3)

    all_results = {}
    for q in queries:
        results = parent_hybrid_retrieve(
            q,
            child_vector_store=vector_store,
            child_bm25=bm25,
            child_chunks=chunks,
            parent_docs=parent_docs,
            top_k=10
        )
        for item in results:
            key = item["doc"].page_content
            if key not in all_results:
                all_results[key] = item
            else:
                all_results[key]["score"] += item["score"]

    docs = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:10]
    return {"docs": docs}


def web_retrieve_node(state: RAGState):
    """
    节点 3b：网页搜索。
    直接用改写后的问题搜互联网。
    """
    docs = web_search(state["rewrite_question"], k=5)
    return {"docs": docs}


def rerank_compress_node(state: RAGState):
    """
    节点 4：精排 + 上下文压缩。
    1. 从 docs 中抽出裸 Document 列表传给 rerank()
    2. 取 top-3 做 Context Compression
    """
    # 抽出裸 Document（rerank 要求 List[Document] 而非 List[dict]）
    docs_for_rerank = [item["doc"] for item in state["docs"]]

    # CrossEncoder 精排
    reranked = rerank(state["rewrite_question"], docs_for_rerank)
    final_docs = reranked[:3]

    # 上下文压缩
    reranker_instance = get_reranker()
    final_docs = compress_documents(
        state["rewrite_question"],
        final_docs,
        reranker_instance,
        keep_ratio=0.5
    )

    return {"final_docs": final_docs}


def generate_node(state: RAGState, llm):
    """
    节点 5：生成最终回答。
    """
    context = "\n\n".join([
        item["doc"].page_content
        for item in state["final_docs"]
    ])

    chain = answer_prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })

    return {"answer": response.content}


# ========== 4. 路由条件函数 ==========

def decide_route(state: RAGState) -> str:
    """根据 classify_route_node 的分类结果，决定下一条边。"""
    if state["route"] == "web":
        return "web_retrieve_node"
    return "local_retrieve_node"


# ========== 5. 构建图 ==========

def build_rag_graph(llm, vector_store, bm25, chunks, parent_docs, use_hyde=False):
    """
    构建并编译 RAG 工作流图。

    参数是 main.py 初始化好的所有组件，以闭包方式注入到各节点中。
    返回编译后的图对象，调用 .invoke({"question": "..."}) 即可执行。
    """
    graph = StateGraph(RAGState)

    # 添加节点。     add_node(节点名称, 执行函数)；节点名称是边跳转的唯一标识；
    # 需要外部资源（llm、向量库）的节点，用lambda s: func(s, 外部参数)闭包捕获，节点运行时自动带入资源；web 检索节点无外部依赖，不需要 lambda
    graph.add_node("rewrite_query_node", lambda s: rewrite_query_node(s, llm))

    graph.add_node("classify_route_node", lambda s: classify_route_node(s, llm))
    graph.add_node(
        "local_retrieve_node",
        lambda s: local_retrieve_node(s, vector_store, bm25, chunks, parent_docs, llm, use_hyde)
    )
    graph.add_node("web_retrieve_node", web_retrieve_node)
    graph.add_node("rerank_compress_node", rerank_compress_node)
    graph.add_node("generate_node", lambda s: generate_node(s, llm))

    # 添加边。 add_edge(A,B)：无条件从 A 节点流转到 B；
    # 流程固定：起点 → 问题改写 → 路由判断
    # graph.add_edge(START, "rewrite_query_node")
    #
    # graph.add_edge("rewrite_query_node", "hyde_node")  # ← 改写完 → 生成假想答案
    # graph.add_edge("hyde_node", "classify_route_node")

    # 图的边：根据 use_hyde 决定是否经过 hyde_node
    graph.add_edge(START, "rewrite_query_node")
    if use_hyde:
        graph.add_node("hyde_node", lambda s: hyde_node(s, llm))
        graph.add_edge("rewrite_query_node", "hyde_node")
        graph.add_edge("hyde_node", "classify_route_node")
    else:
        graph.add_edge("rewrite_query_node", "classify_route_node")


    # 条件边：根据 route 字段决定走 local 还是 web
    graph.add_conditional_edges(
        "classify_route_node",
        decide_route,
        {
            "local_retrieve_node": "local_retrieve_node",
            "web_retrieve_node": "web_retrieve_node",
        }
    )

    # 两条检索路径汇合到同一个精排节点
    graph.add_edge("local_retrieve_node", "rerank_compress_node")
    graph.add_edge("web_retrieve_node", "rerank_compress_node")

    # 精排后生成回答，结束
    # 无论走本地还是网页检索，执行完全部流向重排压缩节点；实现代码复用；
    # 压缩完成→生成答案→流转 END，整张图终止；
    graph.add_edge("rerank_compress_node", "generate_node")
    graph.add_edge("generate_node", END)

    # graph.compile()：编译图，返回可执行对象；外部调用graph.invoke({"question": "xxx"})一键跑完整链路。
    return graph.compile()