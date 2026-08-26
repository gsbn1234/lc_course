from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_parent_child(docs, child_size=200, parent_size=800, overlap=50):
    """
    同一个文档集按两种粒度切分，并建立 child → parent 的映射。

    参数:
        docs:          LangChain Document 列表（PDF 加载后的原始文档）
        child_size:    小块尺寸（字符数），用于检索
        parent_size:   大块尺寸（字符数），用于最终 context
        overlap:       相邻块之间的重叠字符数

    返回:
        child_docs:    小块 Document 列表（每条自带 parent_id 元数据）
        parent_docs:   大块 Document 列表
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=overlap,
        length_function=len
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=overlap,
        length_function=len
    )

    # 先切大块（parent），每块有唯一 ID
    parent_docs = parent_splitter.split_documents(docs)
    for i, doc in enumerate(parent_docs):
        doc.metadata["parent_id"] = f"parent_{i}"

    # 对大块再切小块（child），继承 parent_id
    child_docs = []
    child_sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=overlap,
        length_function=len
    )
    for parent in parent_docs:
        sub_chunks = child_sub_splitter.split_documents([parent])
        for sub in sub_chunks:
            sub.metadata["parent_id"] = parent.metadata["parent_id"]   #split_documents 要求入参是列表，因此必须写[parent]，不能直接传 parent。
        child_docs.extend(sub_chunks)    # extend摊平，一维此时 child_docs是单个列表，里面是child_docs = [Document, Document, Document, ...]
                                        #append是会变成嵌套 [ [doc1,doc2], [doc3], ... ]
    print(f"Parent chunks: {len(parent_docs)}")
    print(f"Child chunks:  {len(child_docs)}")

    return child_docs, parent_docs