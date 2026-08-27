"""
RAG 评估测试集。

每条包含：
  question:     用户问题
  ground_truth: 标准答案（人写的、最理想的回答）

ground_truth 是评估的"尺子"：
  - Context Recall 用它衡量检索到的文档有没有覆盖答案要点
  - 它不是和 LLM 回答比，是和检索出的 context 比

测试集分两类：
  前 4 题：本地检索题 —— 知识库（docs/ PDF）覆盖的技术概念，测本地 RAG 链路。
           （由原 8 题精简而来：删掉主题重复的，每题代表一个主题，跑得更快）
  最后 1 题：联网检索题 —— 知识库没有、必须走 internet_search，测 Researcher
             "本地没有 → 转联网"的决策 + 联网质量

注意：本地题的 ground_truth 对应知识库真实内容；
联网题的 ground_truth 写成"答案应覆盖 XX 要点"，避免具体事实过时。
"""

EVAL_DATASET = [
    {
        "question": "什么是RAG？",
        "ground_truth": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与大语言模型生成相结合的技术框架。它在生成回答前先从外部知识库检索相关文档，将检索结果与用户问题拼接后送入大模型，从而有效缓解大模型的幻觉问题。"
    },
    {
        "question": "FAISS是什么？有什么特点？",
        "ground_truth": "FAISS（Facebook AI Similarity Search）是Meta开源的高效向量相似性搜索库，支持多种索引结构和加速算法，可以在百万级向量数据中快速找到最近邻，是RAG系统中最常用的本地向量存储方案之一。"
    },
    {
        "question": "BM25算法是什么？和向量检索有什么区别？",
        "ground_truth": "BM25是基于TF-IDF改进的关键词检索算法，通过词频、逆文档频率和文档长度归一化计算相关性。向量检索理解语义但可能遗漏精确关键词，BM25擅长精确关键词匹配但不懂语义，两者互补形成混合检索。"
    },
    {
        "question": "混合检索是什么？为什么比单一检索好？",
        "ground_truth": "混合检索是同时使用向量语义检索和BM25关键词检索的策略。向量检索擅长理解语义，BM25擅长精确关键词匹配，两者融合可以互补各自不足，提高检索的召回率和准确率。"
    },

    # ---------- 联网检索题（本地知识库没有，必须走 internet_search） ----------
    {
        "question": "DeepSeek-R1 相比 DeepSeek-V3 有什么关键改进？",
        "ground_truth": "回答应覆盖 DeepSeek-R1 相对 DeepSeek-V3 的核心改进点，至少包含："
                        "1) R1 是强化学习驱动的推理模型，具备长思维链（CoT）推理能力；"
                        "2) 通过纯强化学习（RL）涌现推理行为，无需大量监督微调；"
                        "3) 开源了蒸馏的小模型版本（如 R1-Distill-Qwen/Llama）；"
                        "4) 在数学、代码、逻辑推理类任务上显著优于 V3 基线；"
                        "5) 训练采用 GRPO 强化学习算法。",
    },
]
