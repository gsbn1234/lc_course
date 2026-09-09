"""
多智能体 RAG 评估脚本（LLM-as-Judge 方案）。

评估对象：multi_agent_graph —— 项目生产在用的多智能体协作图
（Supervisor + Researcher + Writer + Reviewer），
不再评估旧的单图 RAG（router_graph，那只是早期原型）。

原理：不依赖 ragas 等外部库，直接用 DeepSeek 充当"评委"。
对每条测试问题的三个维度分别打分：
  1. Context Recall（上下文召回率）—— 上下文 = Researcher 交给 Writer 的研究材料
  2. Faithfulness（忠实度）
  3. Answer Relevancy（回答相关性）

运行：python evaluate.py
注意：会真实调用 DeepSeek（改写/检索/写作/审核）+ Tavily（联网搜索），有 token 成本。
评估用收紧的轮数上限（搜索 3 轮、审核 1 轮）保证可结束、控成本。
"""

import asyncio
import io
from contextlib import redirect_stdout

# 评估脚本自带进度 print；图运行时的内部日志默认压到 WARNING（只看警告/错误），
# 避免每条问题的 Agent 轨迹刷屏。想排查可临时改成 setup_logging("DEBUG")。
from multi_agent.logging_setup import setup_logging
setup_logging("WARNING")

from multi_agent.loader import load_documents
from multi_agent.parent_splitter import split_parent_child
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
from multi_agent.multi_agent_graph import build_multi_agent_graph, extract_answer
from eval_judge import score_triple
from eval_dataset import EVAL_DATASET


async def main():
    print("正在初始化多智能体 RAG 管线...\n")

    docs = load_documents()
    child_docs, parent_docs = split_parent_child(docs, child_size=200, parent_size=800, overlap=50)
    embeddings = get_embeddings()
    child_vector_store = get_vector_store(child_docs, embeddings)
    child_bm25 = create_bm25(child_docs)
    llm = get_llm()

    # 评估用的图：收紧轮数上限（搜索 3 轮、审核 1 轮），
    # 保证每条问题快速跑完，控制 DeepSeek / Tavily 调用成本。
    graph = build_multi_agent_graph(
        llm, child_vector_store, child_bm25, child_docs, parent_docs,
        max_tool_rounds=3, max_review_rounds=1,
    )

    print("初始化完成。开始评估...\n")

    results = []

    for i, item in enumerate(EVAL_DATASET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(EVAL_DATASET)}] {question[:60]}...")

        # 图里 researcher_tool_node 是 async 节点（为兼容 MCP 异步工具），
        # 所以必须走异步 API ainvoke，同步 invoke 会报 "No synchronous function"。
        # 同时给 thread_id（带 checkpointer 的必要条件，每条问题独立会话）+
        # user_id="eval"（隔离评估会话，不混进真实用户的长久记忆）。
        with redirect_stdout(io.StringIO()):
            result = await graph.ainvoke(
                {"question": question},
                config={"configurable": {"thread_id": f"eval-{i}", "user_id": "eval"}},
            )

        # 最终答案在 writer_messages 里，用 extract_answer 提取
        answer = extract_answer(result)
        # Context Recall 的"上下文" = Researcher 整理后交给 Writer 的研究材料。
        # 这就是 Writer 实际看到的全部资料，和旧图的 final_docs 语义等价。
        contexts_text = result.get("research_results", "")

        # 三项独立打分
        recall_score, faith_score, relevancy_score = score_triple(
            question=question,
            answer=answer,
            contexts=contexts_text,
            ground_truth=ground_truth,
            llm=llm,
        )

        results.append({
            "question": question,
            "context_recall": recall_score,
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
        })

        print(f"  Recall={recall_score:.2f}  Faith={faith_score:.2f}  Relev={relevancy_score:.2f}")

    # ========== 3. 汇总输出 ==========

    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("           RAG 评估结果（LLM-as-Judge）")
    print("=" * 60)
    for name, score in [
        ("Context Recall", avg_recall),
        ("Faithfulness", avg_faith),
        ("Answer Relevancy", avg_relevancy),
    ]:
        bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
        print(f"  {name:20s}  {score:.4f}  {bar}")
    print("=" * 60)
    print()

    # 逐条明细，方便定位哪道题拖后腿
    print("各问题得分明细：")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['question'][:50]}")
        print(f"       Recall={r['context_recall']:.2f}  Faith={r['faithfulness']:.2f}  Relev={r['answer_relevancy']:.2f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
