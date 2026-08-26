from langchain_core.prompts import ChatPromptTemplate



prompt=ChatPromptTemplate.from_template(

"""
你是一个严谨的知识问答助手。

请仅根据参考资料回答问题。

如果资料中没有答案，请明确说明“资料中未找到相关信息”。

参考资料：
{context}

用户问题：
{question}

"""

)