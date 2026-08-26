from langchain_openai import ChatOpenAI

from .config import DEEPSEEK_API_KEY



def get_llm():


    return ChatOpenAI(

        model="deepseek-v4-flash",

        base_url="https://api.deepseek.com",

        api_key=DEEPSEEK_API_KEY,

        temperature=0.3

    )