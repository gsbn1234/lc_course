"""
MCP Server — 领域化工具注册。

架构：server 暴露全部领域工具，客户端（load_mcp_tools(domain)）按领域
白名单过滤后再给 Researcher。这样加新领域只改 server 注册 + registry 映射，
不用动接入链路。

领域工具从 mcp_tools/tools/ 按模块导入：
  - ai_learning.py: GitHub / arXiv / HuggingFace（真实工具）
注意：本地包名用 mcp_tools，不能用 mcp —— mcp 是已安装的 MCP SDK
包名，重名会被 Python 优先 import 本地目录，导致 SDK 崩溃。
"""
import datetime
import os
import sys

from mcp.server import FastMCP

# 确保 mcp_tools/tools/ 可以被导入（直接 python 运行时工作目录是项目根，可省略；
# 显式加保险，避免从别的目录启动时找不到）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_tools.tools import ai_learning  # noqa: E402

mcp = FastMCP("MyTools")


# ========== 基础工具（通用） ==========

@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b


@mcp.tool()
def get_time() -> str:
    """返回当前服务器时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========== AI 学习领域工具 ==========

@mcp.tool()
def search_github_repos(query: str, top_k: int = 5) -> str:
    """按关键字搜索 GitHub 开源仓库，返回 star 数、主要语言、最近更新时间。"""
    return ai_learning.search_github_repos(query, top_k)


@mcp.tool()
def search_arxiv_papers(query: str, top_k: int = 5) -> str:
    """按关键字搜索 arXiv 学术论文，返回标题、摘要、作者、链接。"""
    return ai_learning.search_arxiv_papers(query, top_k)


@mcp.tool()
def search_hf_models(query: str, top_k: int = 5) -> str:
    """按关键字搜索 HuggingFace 模型，返回模型名、任务类型、下载量、点赞数。"""
    return ai_learning.search_hf_models(query, top_k)


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输，等 Client 来连
