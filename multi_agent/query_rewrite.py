from langchain_core.prompts import ChatPromptTemplate



rewrite_prompt = ChatPromptTemplate.from_template(
"""
你是一个专业的搜索优化助手。

你的任务：
把用户的问题改写成适合知识库检索的问题。

要求：

1. 保留原始问题意图
2. 补充专业关键词
3. 输出一个检索问题
4. 不要回答问题


用户问题：

{question}


改写后的检索问题：
"""
)



def rewrite_query(
        question,
        llm
):


    chain = rewrite_prompt | llm


    response = chain.invoke(

        {
            "question":question
        }

    )


    rewritten = response.content.strip()


    print("\n===== Query Rewrite =====")

    print(
        rewritten
    )


    return rewritten