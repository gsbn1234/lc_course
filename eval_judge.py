"""
LLM-as-Judge 公共模块：三个评估指标 + 统一打分入口。

evaluate.py / eval_compare.py 共用，避免三份打分 Prompt 三处重复。

数字解析带容错：LLM 打分偶尔会输出 "0.75。"、"0.8（不错）" 这种带杂质的
内容，直接 float() 会崩。这里用正则提取第一个数字，提取不到给 0.5 兜底。
"""
import re

from langchain_core.prompts import ChatPromptTemplate

# —— 指标 1：Context Recall ——
# 核心问题：检索到的文档，有没有 cover 标准答案里的关键信息？
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


def _parse_score(text: str, fallback: float = 0.5) -> float:
    """从 LLM 输出里提取第一个数字，clamp 到 [0,1]；提取失败返回 fallback。"""
    match = re.search(r"[-+]?\d+\.?\d*", text or "")
    if not match:
        return fallback
    return max(0.0, min(1.0, float(match.group())))


def score_triple(question: str, answer: str, contexts: str, ground_truth: str, llm):
    """对一条 (问题, 回答, 上下文, 标准答案) 打三个维度的分。

    三项各是一次独立 LLM 调用，互不干扰。返回 (recall, faith, relevancy)。
    """
    recall = _parse_score(
        (context_recall_prompt | llm).invoke({
            "ground_truth": ground_truth,
            "contexts": contexts,
        }).content
    )
    faith = _parse_score(
        (faithfulness_prompt | llm).invoke({
            "contexts": contexts,
            "answer": answer,
        }).content
    )
    relevancy = _parse_score(
        (answer_relevancy_prompt | llm).invoke({
            "question": question,
            "answer": answer,
        }).content
    )
    return recall, faith, relevancy
