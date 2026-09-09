import logging

import jieba

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)



def create_bm25(chunks):


    bm25_corpus=[

        chunk.page_content

        for chunk in chunks

    ]


    bm25_tokenized=[

        jieba.lcut(
            text.lower()
        )

        for text in bm25_corpus

    ]


    bm25=BM25Okapi(
        bm25_tokenized
    )


    logger.info("BM 索引建立完成（共 %d 块）", len(chunks))


    return bm25