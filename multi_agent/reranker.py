from sentence_transformers import CrossEncoder
reranker = None


def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder("BAAI/bge-reranker-base")
    return reranker



def rerank(
        question,
        ranked_docs,

):


    reranker = get_reranker()

    pairs=[]


    for doc in ranked_docs:


        pairs.append(

            (
                question,

                doc.page_content

            )

        )



    scores=reranker.predict(
        pairs
    )


    results=[]


    for doc,new_score in zip(

        ranked_docs,

        scores

    ):


        results.append(

            {

            "doc":doc,

            "score":float(new_score)

            }

        )



    results.sort(

        key=lambda x:x["score"],

        reverse=True

    )

    # for i,item in enumerate(results):
    #     print(f"top{i+1}")
    #     print(f"CrossEncoder:{item["score"]:.4f}")
    #     print(item["doc"])
    #     print()

    # 精简：一行概括 top-3 分数
    top_scores = ", ".join(
        f"{r['score']:.2f}" for r in results[:3]
    )
    print(f"[Rerank] top-3 scores: {top_scores}")

    return results

