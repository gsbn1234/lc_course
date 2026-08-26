from langchain_community.document_loaders import PyPDFDirectoryLoader

from .config import PDF_PATH



def load_documents():

    loader = PyPDFDirectoryLoader(
        PDF_PATH
    )

    docs = loader.load()

    print(
        f"加载文档页数：{len(docs)}"
    )

    return docs