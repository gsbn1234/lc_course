from langchain_core.prompts import ChatPromptTemplate



multi_query_prompt = ChatPromptTemplate.from_template(
"""
你是一个搜索优化专家。

你的任务：

根据用户问题生成多个不同角度的检索问题。

要求：

1. 保留用户核心意图
2. 使用不同表达方式
3. 增加专业关键词
4. 只输出问题，每行一个
5. 不要回答


用户问题：

{question}


生成的问题：
"""
)



def generate_queries(
        question,
        llm,
        num_queries=4
):


    chain = multi_query_prompt | llm


    response = chain.invoke(

        {
            "question":question
        }

    )


    content=response.content.strip()



    queries=[
        q.strip()
        for q in content.split("\n")
        if q.strip()
    ]



    # 防止模型输出编号
    clean_queries=[]


    for q in queries:

        q=q.lstrip(
            "123456789.-、 "
        )

        clean_queries.append(q)



    # 保留数量

    queries=clean_queries[:num_queries]


    print("\n===== Multi Query =====")

    for i,q in enumerate(
        queries,
        1
    ):
        print(
            f"{i}. {q}"
        )


    return queries