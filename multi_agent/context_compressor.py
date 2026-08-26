import re            #导入正则表达式库，用于文本分句
from copy import copy      #导入浅拷贝函数，用来复制 Document 文档对象，防止修改原始文档。


def split_sentences(text):
    """
    将中文文本按句号、问号、感叹号、分号、换行切分为句子。
    用 look-behind 保留分隔符，避免句子末尾标点丢失。  也就是?<=xxx
    """
    sentences = re.split(r'(?<=[。！？；\n])', text)  #只匹配「标点符号的后一个位置」，不会把标点本身当作切割符删掉
    return [s.strip() for s in sentences if s.strip()]   #返回清洗干净的句子列表。


def compress_documents(query, docs, reranker, keep_ratio=0.5):
    """
    上下文压缩：对每个文档逐句打分，只保留与问题最相关的句子。

    工作流程：
      1. 将每个文档拆成句子
      2. 用 CrossEncoder 对每个 (query, sentence) 对打分
      3. 按相关性排序，保留前 keep_ratio 比例的句子
      4. 按原文顺序重建压缩后的文本（保证可读性）

    参数：
        query:      用户问题（建议用 rewrite 后的问题，表述更精准）
        docs:       从 rerank 返回的文档列表 [{"doc": Document, "score": float}, ...]
        reranker:   CrossEncoder 实例（复用 reranker.py 中已有的 BAAI/bge-reranker-base）
        keep_ratio: 保留句子的比例，默认 0.5（保留一半）

    返回：
        压缩后的文档列表，结构不变，但 page_content 只包含相关句子
    """
    compressed = []

    for item in docs:
        doc = item["doc"]
        score = item["score"]
        text = doc.page_content

        # 拆句子
        sentences = split_sentences(text)

        # 句子太少的不压缩，原样保留
        if len(sentences) <= 2:
            compressed.append(item)
            continue

        # 对每个句子打相关性分
        pairs = [(query, sent) for sent in sentences]
        sentence_scores = reranker.predict(pairs)

        # 配对 (句子, 分数)，按分数降序
        scored = list(zip(sentences, sentence_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 取前 keep_ratio 比例，至少保留 1 句
        keep_count = max(1, int(len(sentences) * keep_ratio))
        kept = scored[:keep_count]

        # 按原文顺序重建，保持可读性
        kept.sort(key=lambda x: sentences.index(x[0]))
        compressed_text = "".join([s for s, _ in kept])

        # 创建压缩后的文档（浅拷贝，不破坏原对象）
        new_doc = copy(doc)
        new_doc.page_content = compressed_text

        compressed.append({
            "doc": new_doc,
            "score": score
        })

        # print(f"\n[Context Compression] 原 {len(sentences)} 句 → 保留 {keep_count} 句")
        # print(f"  压缩前 {len(text)} 字 → 压缩后 {len(compressed_text)} 字")
    print(f"[Compress] {len(compressed)} docs compressed")
    return compressed