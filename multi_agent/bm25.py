import jieba

from rank_bm25 import BM25Okapi



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


    print(
        "BM索引建立完成"
    )


    return bm25