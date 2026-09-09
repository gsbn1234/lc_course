"""
三档基线对比评估：无检索直答 vs 单路检索 vs 完整系统。

运行：python eval_baseline.py
产出：控制台对比表 + docs/eval_baseline_result.md（可直接贴进 README）

为什么要有这个脚本：
  eval_compare.py 只回答"HyDE 开/关"这一个组件问题；
  本脚本回答系统级问题——"有没有 RAG 值不值、你的检索链路比 Naive 好在哪"。

三档系统（同一份测试集、同一个 LLM 裁判、同一套 R/F/A 打分，只换检索链路）：
  [档位1] 无检索直答 —— 问题直接丢给 LLM，零上下文。
          → 证明"为什么需要 RAG"：Context Recall 必为 0，Faithfulness 必然低。
  [档位2] 单路检索   —— 只做向量 top-k（200 字 child 块直接拼），
          不做父子切分/改写/HyDE/多查询/精排/压缩。
          → 证明"高级组件为什么存在"：上下文碎片化，F/A 低于完整系统。
  [档位3] 完整系统   —— 改写 + 路由 + 父子混合检索 + HyDE + 多查询 + 精排 + 压缩，
          即 README 架构图里的检索流水线。
"""
import io
import time
from contextlib import redirect_stdout
from datetime import date

from multi_agent.logging_setup import setup_logging
setup_logging("WARNING")

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from multi_agent.config import PROJECT_ROOT
from multi_agent.loader import load_documents
from multi_agent.parent_splitter import split_parent_child
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
from multi_agent.router_graph import build_rag_graph
from multi_agent.prompt import prompt as answer_prompt
from eval_dataset import EVAL_DATASET

# ========== 三个评估 Prompt（和 eval_compare.py 完全一致，保证口径统一） ==========

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


# ========== 性能/成本采集 ==========

# DeepSeek V4-Flash 官网现行计价（元 / 百万 token）。评估为一次性脚本，基本无缓存命中，按 miss 计。
INPUT_PRICE_PER_M = 1.0    # 输入（缓存未命中）
OUTPUT_PRICE_PER_M = 2.0   # 输出


class TokenCounter(BaseCallbackHandler):
    """挂在 LLM 调用链上，累计所有经手调用的输入/输出 token。

    用法：把 counter 放进 config={"callbacks": [counter]}，然后对每次调用前后
    快照 self.input_tokens / self.output_tokens 即可拿到"这一段的 token 增量"。
    """

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        for gen in response.generations:
            um = getattr(gen[0].message, "usage_metadata", None) or {}
            # langchain 归一化字段；不同 provider 可能有 prompt_tokens 命名
            self.input_tokens += um.get("input_tokens") or um.get("prompt_tokens") or 0
            self.output_tokens += um.get("output_tokens") or um.get("completion_tokens") or 0
        # 兜底：OpenAI 兼容的 llm_output.token_usage（usage_metadata 缺失时）
        lu = (response.llm_output or {}).get("token_usage") or {}
        self.input_tokens += lu.get("prompt_tokens", 0)
        self.output_tokens += lu.get("completion_tokens", 0)


# ========== 统一评估入口 ==========

def run_evaluation(name, answer_fn, llm):
    """
    跑一轮评估：answer_fn(question, callbacks=None) -> (answer, contexts_text)。
    三个档位共用同一接口、同一套 judge、同一份数据集，
    保证打分口径一致，只有"检索链路"不同。

    性能采集：每题记答案阶段耗时 + 输入/输出 token（只计答案的 LLM 调用）；
    tier_total_* 把裁判打分也计入，用于算"跑一次评估烧多少钱"。
    """
    results = []
    perf_rows = []
    tier_total_in = 0
    tier_total_out = 0
    for i, item in enumerate(EVAL_DATASET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        # 答案阶段：计时 + 只数这一段的 token
        counter = TokenCounter()
        t0 = time.perf_counter()
        answer, contexts_text = answer_fn(question, callbacks=[counter])
        latency_s = time.perf_counter() - t0
        a_in, a_out = counter.input_tokens, counter.output_tokens

        # 裁判阶段：3 次打分 LLM 调用，沿用同一计数器（计入总成本，不单独计时）
        recall_score = float(
            (context_recall_prompt | llm).invoke(
                {"ground_truth": ground_truth, "contexts": contexts_text},
                config={"callbacks": [counter]},
            ).content.strip()
        )
        faith_score = float(
            (faithfulness_prompt | llm).invoke(
                {"contexts": contexts_text, "answer": answer},
                config={"callbacks": [counter]},
            ).content.strip()
        )
        relevancy_score = float(
            (answer_relevancy_prompt | llm).invoke(
                {"question": question, "answer": answer},
                config={"callbacks": [counter]},
            ).content.strip()
        )

        tier_total_in += counter.input_tokens
        tier_total_out += counter.output_tokens
        perf_rows.append({"latency_s": latency_s, "in_tokens": a_in, "out_tokens": a_out})
        results.append({
            "question": question,
            "context_recall": recall_score,
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
        })
        print(f"  [{i}/{len(EVAL_DATASET)}] {question[:22]:<22s}"
              f" R={recall_score:.2f} F={faith_score:.2f} A={relevancy_score:.2f}"
              f" | {latency_s:.1f}s {a_in}in {a_out}out")

    n = len(results)
    avg_recall = sum(r["context_recall"] for r in results) / n
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / n
    perf_agg = {
        "avg_latency_s": sum(p["latency_s"] for p in perf_rows) / n,
        "avg_in": sum(p["in_tokens"] for p in perf_rows) / n,
        "avg_out": sum(p["out_tokens"] for p in perf_rows) / n,
        "tier_in": tier_total_in,      # 含裁判，用于总成本
        "tier_out": tier_total_out,
    }
    return results, avg_recall, avg_faith, avg_relevancy, perf_agg


# ========== 三个档位的 answer_fn ==========

direct_prompt = ChatPromptTemplate.from_template(
    "你是知识问答助手。直接回答下面的问题（不要提及任何检索过程）：\n问题：{question}"
)


def direct_answer(question, llm, callbacks=None):
    """档位1：无检索直答。上下文为空，测 LLM 裸答质量。"""
    answer = (direct_prompt | llm).invoke(
        {"question": question}, config={"callbacks": callbacks}
    ).content
    return answer, ""


def naive_rag_answer(question, llm, vector_store, callbacks=None):
    """档位2：单路检索。向量 top-k 直接拼上下文，不做任何增强。"""
    docs = vector_store.similarity_search(question, k=4)
    contexts_text = "\n\n".join(d.page_content for d in docs)
    answer = (answer_prompt | llm).invoke(
        {"context": contexts_text, "question": question},
        config={"callbacks": callbacks},
    ).content
    return answer, contexts_text


def full_answer_fn(graph):
    """档位3：完整系统。跑整张图，从 final_docs 取上下文。"""
    def _fn(question, callbacks=None):
        config = {"callbacks": callbacks} if callbacks else {}
        with redirect_stdout(io.StringIO()):
            result = graph.invoke({"question": question}, config=config)
        contexts_text = "\n---\n".join(
            item["doc"].page_content for item in result["final_docs"]
        )
        return result["answer"], contexts_text
    return _fn


# ========== 主流程 ==========

def main():
    print("正在初始化 RAG 管线...\n")
    docs = load_documents()
    child_docs, parent_docs = split_parent_child(docs, child_size=200, parent_size=800, overlap=50)
    embeddings = get_embeddings()
    child_vector_store = get_vector_store(child_docs, embeddings)
    child_bm25 = create_bm25(child_docs)
    llm = get_llm()

    # 档位3：完整系统（HyDE 开）
    graph_full = build_rag_graph(
        llm, child_vector_store, child_bm25, child_docs, parent_docs, use_hyde=True
    )

    tiers = [
        ("无检索直答", lambda q, callbacks=None: direct_answer(q, llm, callbacks=callbacks)),
        ("单路检索",   lambda q, callbacks=None: naive_rag_answer(q, llm, child_vector_store, callbacks=callbacks)),
        ("完整系统",   full_answer_fn(graph_full)),
    ]

    all_results = {}
    run_total_in = 0
    run_total_out = 0
    for name, fn in tiers:
        print(f"\n========== 档位：{name} ==========")
        results, r, f, a, perf = run_evaluation(name, fn, llm)
        all_results[name] = (results, r, f, a, perf)
        run_total_in += perf["tier_in"]
        run_total_out += perf["tier_out"]

    # ========== 控制台对比 ==========
    print("\n" + "=" * 78)
    print("                    三档基线对比")
    print("=" * 78)
    print(f"  {'档位':<10s} {'Recall':>8s} {'Faith':>8s} {'Relev':>8s} {'耗时/题s':>9s}")
    for name, (_, r, f, a, perf) in all_results.items():
        print(f"  {name:<10s} {r:>8.4f} {f:>8.4f} {a:>8.4f} {perf['avg_latency_s']:>9.2f}")
    print("=" * 78)
    cost_yuan = (run_total_in * INPUT_PRICE_PER_M + run_total_out * OUTPUT_PRICE_PER_M) / 1_000_000
    print(f"\n  全程 token：{run_total_in / 1000:.1f}k 输入 + {run_total_out / 1000:.1f}k 输出"
          f" ≈ ¥{cost_yuan:.3f}（输入 ¥1/百万·miss + 输出 ¥2/百万）")

    # ========== 写 Markdown ==========
    out_path = PROJECT_ROOT / "docs" / "eval_baseline_result.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(build_markdown(all_results, run_total_in, run_total_out), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


def build_markdown(all_results, run_total_in, run_total_out):
    lines = []
    lines.append("# 三档基线对比评估结果")
    lines.append("")
    lines.append(f"> 生成时间：{date.today().isoformat()}")
    lines.append("> 测试集：自建 5 题（docs/ 覆盖的 4 个本地题 + 1 个联网题）")
    lines.append("> 裁判：DeepSeek LLM-as-Judge（Context Recall / Faithfulness / Answer Relevancy，0.0-1.0）")
    lines.append("> 口径：同一份测试集、同一个裁判、同一套打分规则，仅检索链路不同。")
    lines.append("")
    lines.append("## 总平均")
    lines.append("")
    lines.append("| 档位 | Context Recall | Faithfulness | Answer Relevancy |")
    lines.append("|------|---------------|--------------|------------------|")
    for name, (_, r, f, a, _) in all_results.items():
        lines.append(f"| {name} | {r:.4f} | {f:.4f} | {a:.4f} |")
    lines.append("")
    lines.append("## 性能与成本")
    lines.append("")
    lines.append(f"> 实测于 {date.today().isoformat()}，模型 deepseek-v4-flash。")
    lines.append("> 耗时 = 答案阶段墙钟时间；token = 答案阶段 LLM 输入/输出（裁判打分只计入总成本）。")
    lines.append("")
    lines.append("| 档位 | 平均耗时/题 (s) | 平均输入 token | 平均输出 token |")
    lines.append("|------|---------------|---------------|---------------|")
    for name, (_, _, _, _, perf) in all_results.items():
        lines.append(
            f"| {name} | {perf['avg_latency_s']:.2f} | {perf['avg_in']:.0f} | {perf['avg_out']:.0f} |"
        )
    cost_yuan = (run_total_in * INPUT_PRICE_PER_M + run_total_out * OUTPUT_PRICE_PER_M) / 1_000_000
    lines.append("")
    lines.append(
        f"- 一次完整评估（3 档 × 5 题 + 15 次裁判打分）共消耗约 "
        f"{run_total_in / 1000:.1f}k 输入 + {run_total_out / 1000:.1f}k 输出 token，"
        f"按官网价（输入 ¥1/百万·缓存未命中、输出 ¥2/百万）约 **¥{cost_yuan:.3f}**。"
    )
    lines.append("")
    lines.append("## 每题明细")
    lines.append("")
    lines.append("| # | 问题 | 档位 | R | F | A |")
    lines.append("|---|------|------|---|---|---|")
    for name, (results, _, _, _, _) in all_results.items():
        for i, res in enumerate(results, 1):
            lines.append(
                f"| {i} | {res['question'][:30]} | {name} "
                f"| {res['context_recall']:.2f} | {res['faithfulness']:.2f} | {res['answer_relevancy']:.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
