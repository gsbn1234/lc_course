import logging

from langchain_community.document_loaders import PyPDFDirectoryLoader

from .config import PDF_PATH

logger = logging.getLogger(__name__)



def load_documents():

    loader = PyPDFDirectoryLoader(
        PDF_PATH
    )

    docs = loader.load()

    logger.info("加载文档页数：%d", len(docs))

    return docs