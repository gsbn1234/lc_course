



from multi_agent.logging_setup import setup_logging
setup_logging()

from multi_agent.loader import load_documents
from multi_agent.parent_splitter import split_parent_child   # ← 替换原来的 splitter
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
import asyncio
import uuid



# 1. 加载原始文档
docs = load_documents()

# 2. 父子切分（替换原来的 split_documents）
child_docs, parent_docs = split_parent_child(
    docs,
    child_size=200,    # 小块 200 字符，搜得精
    parent_size=800,   # 大块 800 字符，上下文完整
    overlap=50
)

# 3. 在小块上建索引（搜得准）
embeddings = get_embeddings()
child_vector_store = get_vector_store(child_docs, embeddings)
child_bm25 = create_bm25(child_docs)

# 4. LLM
llm = get_llm()




# 5. 构建 Tool-Using Agent（ReAct 模式）

from multi_agent.multi_agent_graph import build_multi_agent_graph, extract_answer

graph = build_multi_agent_graph(
    llm,
    child_vector_store,
    child_bm25,
    child_docs,
    parent_docs,
    max_tool_rounds=5,
)


# 6. 测试
questions = [
    "什么是RAG？",
    "2026年7月人工智能领域有什么重大新闻？",
]

async def ask(graph, question):
    # 图中 researcher_tool_node 是 async 节点（为兼容 MCP 纯异步工具），
    # 必须用异步 API（ainvoke）跑图；同步 invoke 会报
    # "No synchronous function provided to researcher_tool_node"。
    return await graph.ainvoke(
        {
            "question": question,
            "researcher_rounds": 0,
            "max_tool_rounds": 5,
        },
        config={
            "configurable": {"thread_id": f"main_{uuid.uuid4().hex[:8]}"},
            # 下面三个键是给 LangSmith 看的：让每条轨迹更好认、更好筛。
            # 开了追踪后，到 smith.langchain.com 就能按 run_name / tag 过滤这次跑图。
            "run_name": "multi_agent_rag",    # 轨迹名（整条 trace 的标题）
            "tags": ["main", "learning"],      # 标签：可按场景再分类
            "metadata": {"source": "main.py"}, # 自定义元数据，任意键值对
        },
    )


for q in questions:
    print("\n" + "=" * 60)
    print(f"问题: {q}")
    print("=" * 60)

    result = asyncio.run(ask(graph, q))

    answer = extract_answer(result)
    print(f"\n回答:\n{answer}")
    print(f"\nResearcher 搜索轮数: {result.get('researcher_rounds', 0)}")