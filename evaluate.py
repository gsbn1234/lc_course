"""
RAG 管线评估脚本（LLM-as-Judge 方案）。

原理：不依赖 ragas 等外部库，直接用 DeepSeek 充当"评委"。
对每条测试问题的三个维度分别打分：
  1. Context Recall（上下文召回率）
  2. Faithfulness（忠实度）
  3. Answer Relevancy（回答相关性）

输出：每个指标的平均分 + 逐条明细。

运行：python evaluate.py
"""

from langchain_core.prompts import ChatPromptTemplate
import io
from contextlib import redirect_stdout



# ========== 0. 初始化 RAG 管线（和 main.py 一样） ==========

from multi_agent.loader import load_documents
from multi_agent.parent_splitter import split_parent_child
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
from multi_agent.router_graph import build_rag_graph
from eval_dataset import EVAL_DATASET

print("正在初始化 RAG 管线...\n")

docs = load_documents()
child_docs, parent_docs = split_parent_child(docs, child_size=200, parent_size=800, overlap=50)
embeddings = get_embeddings()
child_vector_store = get_vector_store(child_docs, embeddings)
child_bm25 = create_bm25(child_docs)
llm = get_llm()

graph = build_rag_graph(llm, child_vector_store, child_bm25, child_docs, parent_docs)

print("初始化完成。开始评估...\n")


# ========== 1. 三个评估 Prompt ==========

# —— 指标 1：Context Recall ——
# 核心问题：检索到的文档，有没有 cover 标准答案里的关键信息？
# 即使 LLM 最终回答得很好，如果 context 里根本没相关内容，那说明检索层有问题。
context_recall_prompt = ChatPromptTemplate.from_template("""
你是一个严格的内容评估专家。

你的任务：判断检索到的上下文文档是否覆盖了标准答案中的关键信息。

标准答案（理想情况下应有这些内容）：
{ground_truth}

实际检索到的上下文文档（RAG 系统从知识库里搜出来的）：
{contexts}

打分标准：
- 1.0：上下文完整覆盖了标准答案的所有关键信息点
- 0.7：覆盖了大部分关键信息，但有 1-2 处遗漏
- 0.5：覆盖了一半左右的关键信息
- 0.3：只覆盖了少量信息
- 0.0：完全没有覆盖

只输出一个 0.0 到 1.0 之间的数字，不要任何解释。
例如：0.75""")


# —— 指标 2：Faithfulness ——
# 核心问题：LLM 有没有编造上下文里不存在的东西？
# 即使 context 很全，LLM 也可能自己脑补训练数据里的内容。
faithfulness_prompt = ChatPromptTemplate.from_template("""
你是一个严格的事实核查专家。

你的任务：检查 LLM 的回答是否完全基于给定的上下文文档，有没有编造。

上下文文档（LLM 生成回答时能看到的全部资料）：
{contexts}

LLM 生成的回答：
{answer}

规则：
- 回答中的每一项事实主张，都必须在上下文中找到依据
- 任何上下文没有提及的事実、数字、人名、时间、技术细节，都算"编造"
- 每出现一处编造就大幅扣分

打分标准：
- 1.0：所有事实主张都能在上下文中找到明确依据
- 0.7：有 1 处轻微润色或无关紧要的添加
- 0.5：有 2-3 处上下文不支持的表述
- 0.3：明显有编造内容
- 0.0：回答完全是凭空编造

只输出一个 0.0 到 1.0 之间的数字，不要任何解释。
例如：0.90""")


# —— 指标 3：Answer Relevancy ——
# 核心问题：回答有没有跑题？
# 回答可能 faithful（没编造），但答非所问。
answer_relevancy_prompt = ChatPromptTemplate.from_template("""
你是一个严谨的语义评估专家。

你的任务：判断 LLM 的回答是否直接、完整地回应了用户问题。

用户问题：
{question}

LLM 生成的回答：
{answer}

规则：
- 回答是否直接针对问题？还是绕来绕去说废话？
- 回答是否完整？有没有遗漏问题的核心部分？
- 回答里有没有大量和问题无关的内容？

打分标准：
- 1.0：直接、精准、完整地回答了问题
- 0.7：大体回答了，有少量不切题或不够完整
- 0.5：回答了一半，有较多跑题或遗漏
- 0.3：勉强沾边，大部分不相关
- 0.0：完全答非所问

只输出一个 0.0 到 1.0 之间的数字，不要任何解释。
例如：0.85""")


# ========== 2. 逐条跑评估 ==========

results = []

for i, item in enumerate(EVAL_DATASET, 1):
    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"[{i}/{len(EVAL_DATASET)}] {question[:60]}...")

    # 2a. 跑 RAG 管线，拿到回答和检索到的文档
    # result = graph.invoke({"question": question})
    # answer = result["answer"]

    # 2a. 跑 RAG 管线，静默所有模块内部 print
    with redirect_stdout(io.StringIO()):
        result = graph.invoke({"question": question})
    answer = result["answer"]

    # 把 final_docs 拼成 text 给评估 prompt 用
    contexts_text = "\n---\n".join([
        d["doc"].page_content for d in result["final_docs"]
    ])

    # 2b. 三项独立打分
    #    每一项都是独立一次 LLM 调用，互不干扰

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