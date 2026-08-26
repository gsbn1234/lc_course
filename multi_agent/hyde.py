"""
HyDE（Hypothetical Document Embeddings，假想文档检索）

原理：
  用户问题 → LLM 编一段假答案 → 用假答案的文本去向量检索

为什么有效：
  问题的 Embedding 是"疑问句向量"，文档的 Embedding 是"陈述句向量"，
  两者在向量空间里有天然间距。假答案填平了这个沟：
  - 假答案的文体 = 陈述句（和真实文档一致）
  - 假答案的内容 = 围绕用户意图（和问题相关）

  即使假答案有事实错误也没关系——关键是风格和意图对齐，
  Embedding 算的是语义方向而非事实准确度。
"""

from langchain_core.prompts import ChatPromptTemplate

hyde_prompt = ChatPromptTemplate.from_template("""
你是一个专业的技术文档写作者。

请根据以下问题，用技术文档的口吻写一段回答。即使你不确定具体内容，
也请基于你的专业知识写一段看起来合理的、信息密集的解释。

要求：
- 使用陈述句、专业术语、信息密度高的写法
- 像百科词条或技术白皮书那样写
- 长度约 150-250 字
- 直接写内容，不要以"根据我的了解""据我所知"等开头

问题：{question}

回答：""")


def generate_hypothetical_answer(rewrite_question: str, llm) -> str:
    """
    生成假想答案。

    参数:
        question: 用户问题（建议用 rewrite_question，表达更精准）
        llm:      LLM 实例

    返回:
        hyde_answer: 一段看起来像文档的假想回答（纯文本）
    """
    chain = hyde_prompt | llm
    response = chain.invoke({"question": rewrite_question})
    hyde_answer = response.content.strip()
    print(f"\n[HyDE] 假想答案已生成（{len(hyde_answer)} 字）: {hyde_answer[:80]}...")
    return hyde_answer