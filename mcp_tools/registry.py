"""
MCP 工具领域注册表 —— 工具按"知识库领域"组织，而不是按"项目"写死。

核心思想：项目是通用 RAG 底座，知识库（上传的 PDF）决定领域，
工具集跟着领域走。上传时选择领域，Researcher 只拿到该领域的工具，
不会被无关工具干扰 LLM 的决策。

当前领域：
  ai_learning   AI 学习/技术文档 —— 工具：GitHub 查开源项目、arXiv 查论文、
                HuggingFace 查模型（本项目的现成知识库 + 主推领域）
  general       通用 —— 无专用工具，走 Tavily 网页搜索 + 本地知识库就够
  finance       金融 —— 预留（接行情/汇率/公告时填工具名）
  medical       医疗 —— 预留（接文献库/药品库时填工具名）

新增领域步骤：
  1. 在 mcp_tools/tools/ 下建 <domain>.py，写工具函数
  2. 在 my_mcp_server.py 里注册
  3. 在下面映射表加一行  domain -> [工具名...]
  4. 前端 app.py 的领域下拉框加选项
"""

# 领域 → 暴露给 Researcher 的工具名（白名单）
DOMAIN_TOOLS = {
    "ai_learning": [
        "search_github_repos",
        "search_arxiv_papers",
        "search_hf_models",
    ],
    "general": [],      # 通用领域不接专用工具
    "finance": [],      # 预留
    "medical": [],      # 预留
}

# 领域中文名（给前端下拉框显示用）
DOMAIN_LABELS = {
    "ai_learning": "AI 学习（技术文档）",
    "general": "通用",
    "finance": "金融（预留）",
    "medical": "医疗（预留）",
}


def get_domain_tools(domain: str) -> list[str]:
    """返回某领域允许使用的工具名列表；未知领域返回空（降级为纯本地工具）。"""
    return DOMAIN_TOOLS.get(domain, [])
