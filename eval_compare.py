"""
HyDE 对比评估：关 vs 开，量化检索质量变化。

运行：python eval_compare.py
"""
import io
from contextlib import redirect_stdout

from multi_agent.logging_setup import setup_logging
setup_logging("WARNING")

from langchain_core.prompts import ChatPromptTemplate

from multi_agent.loader import load_documents
from multi_agent.parent_splitter import split_parent_child
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
from multi_agent.router_graph import build_rag_graph
from eval_dataset import EVAL_DATASET

# ========== 三个评估 Prompt（和 evaluate.py 一样） ==========

context_recall_prompt = ChatPromptTemplate.from_template("""
你是一个严格的内容评估专家。
你的任务：判断检索到的上下文文档是否覆盖了标准答案中的关键信息。
标准答案：{ground_truth}
实际检索到的上下文文档：{contexts}
打分标准：
- 1.0：上下文完整覆盖了标准答案的所有关键信息点
- 0.7：覆盖了大部分关键信息，但有 1-2 处遗漏
- 0.5：覆盖了一半左右的关键信息
- 0.3：只覆盖了少量信息
- 0.0：完全没有覆盖
只输出一个 0.0 到 1.0 之间的数字，不要任何解释。""")

faithfulness_prompt = ChatPromptTemplate.from_template("""
你是一个严格的事实核查专家。
你的任务：检查 LLM 的回答是否完全基于给定的上下文文档，有没有编造。
上下文文档：{contexts}
LLM 生成的回答：{answer}
规则：回答中的每一项事实主张，都必须在上下文中找到依据。
打分标准：
- 1.0：所有事实主张都能在上下文中找到明确依据
- 0.7：有 1 处轻微润色或无关紧要的添加
- 0.5：有 2-3 处上下文不支持的表述
- 0.3：明显有编造内容
- 0.0：回答完全是凭空编造
只输出一个 0.0 到 1.0 之间的数字，不要任何解释。""")

answer_relevancy_prompt = ChatPromptTemplate.from_template("""
你是一个严谨的语义评估专家。
你的任务：判断 LLM 的回答是否直接、完整地回应了用户问题。
用户问题：{question}
LLM 生成的回答：{answer}
打分标准：
- 1.0：直接、精准、完整地回答了问题
- 0.7：大体回答了，有少量不切题或不够完整
- 0.5：回答了一半，有较多跑题或遗漏
- 0.3：勉强沾边，大部分不相关
- 0.0：完全答非所问
只输出一个 0.0 到 1.0 之间的数字，不要任何解释。""")


def run_evaluation(name, graph, llm):
    """跑一轮评估，返回 results 列表和三个平均分。"""
    results = []
    for i, item in enumerate(EVAL_DATASET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i}/{len(EVAL_DATASET)}] {question[:50]}...", end=" ") #end=" " 不让换行。

        with redirect_stdout(io.StringIO()):
            result = graph.invoke({"question": question})

        answer = result["answer"]
        contexts_text = "\n---\n".join([
            d["doc"].page_content for d in result["final_docs"]
        ])

        recall_score = float(
            (context_recall_prompt | llm).invoke({
                "ground_truth": ground_truth,
                "contexts": contexts_text,
            }).content.strip()
        )
        faith_score = float(
            (faithfulness_prompt | llm).invoke({
                "contexts": contexts_text,
                "answer": answer,
            }).content.strip()
        )
        relevancy_score = float(
            (answer_relevancy_prompt | llm).invoke({
                "question": question,
                "answer": answer,
            }).content.strip()
        )

        results.append({
            "question": question,
            "context_recall": recall_score,
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
        })
        print(f"R={recall_score:.2f} F={faith_score:.2f} A={relevancy_score:.2f}")

    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    return results, avg_recall, avg_faith, avg_relevancy


# ========== 主流程 ==========

print("正在初始化 RAG 管线...\n")
docs = load_documents()
child_docs, parent_docs = split_parent_child(docs, child_size=200, parent_size=800, overlap=50)
embeddings = get_embeddings()
child_vector_store = get_vector_store(child_docs, embeddings)
child_bm25 = create_bm25(child_docs)
llm = get_llm()

# 构建两个图：一个不用 HyDE，一个用 HyDE
graph_baseline = build_rag_graph(llm, child_vector_store, child_bm25, child_docs, parent_docs, use_hyde=False)
graph_hyde    = build_rag_graph(llm, child_vector_store, child_bm25, child_docs, parent_docs, use_hyde=True)

# 跑两轮
print("\n========== 第一轮：基线（无 HyDE） ==========")
results_baseline, r1, f1, a1 = run_evaluation("baseline", graph_baseline, llm)

print("\n========== 第二轮：HyDE ==========")
results_hyde, r2, f2, a2 = run_evaluation("hyde", graph_hyde, llm)

# 对比输出
print("\n" + "=" * 70)
print("                    对比结果")
print("=" * 70)
print(f"  {'指标':<22s} {'无 HyDE':>8s} {'有 HyDE':>8s} {'变化':>10s}")
print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*10}")
for name, before, after in [
    ("Context Recall",    r1, r2),
    ("Faithfulness",      f1, f2),
    ("Answer Relevancy",  a1, a2),
]:
    diff = after - before
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    print(f"  {name:<22s} {before:>8.4f} {after:>8.4f} {arrow} {abs(diff):>8.4f}")
print("=" * 70)