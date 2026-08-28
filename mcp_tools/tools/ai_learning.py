"""
AI 学习领域的真实 MCP 工具。

给 Researcher 补充三类 Tavily 网页搜索覆盖不好的"结构化一手数据"：
  1. search_github_repos  —— 开源项目技术选型（star/语言/活跃度）
  2. search_arxiv_papers  —— 学术论文（AI 前沿一手信息）
  3. search_hf_models     —— 模型选型（参数规模/任务/下载量）

企业级要点：
  - 每个工具带超时（httpx.Timeout），避免外部 API 卡死拖住 Researcher
  - 失败不抛异常，返回可读错误串（LLM 能理解"这次查不到"），并打印日志
  - 返回的是 LLM 友好的结构化文本，不是裸 JSON
"""
import os

import httpx

# 全局统一的超时配置（秒）：外部 API 最慢 8 秒必须给结果
_TIMEOUT = httpx.Timeout(8.0)

# GitHub 匿名限流 60 次/时，够学习用；配了 GITHUB_TOKEN 就带上提额到 5000
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def search_github_repos(query: str, top_k: int = 5) -> str:
    """按关键字搜索 GitHub 开源仓库，返回 star 数、主要语言、最近更新时间。

    适合：技术选型对比（"哪个 RAG 框架最火"）、找参考实现、调研开源项目。

    Args:
        query: 搜索关键字，如 "RAG framework"
        top_k: 返回条数，默认 5，最大 10
    """
    headers = {"Accept": "application/vnd.github+json"}
    if _GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {_GITHUB_TOKEN}"

    try:
        resp = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": min(top_k, 10)},
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except httpx.TimeoutException:
        print("[MCP][GitHub] 请求超时")
        return "GitHub 搜索超时，请稍后重试或改用网页搜索。"
    except Exception as e:
        print(f"[MCP][GitHub] 搜索失败：{e}")
        return f"GitHub 搜索失败：{type(e).__name__}: {e}"

    if not items:
        return "GitHub 上没有找到相关仓库。"

    lines = [f"GitHub 搜索结果（按 star 排序）："]
    for i, r in enumerate(items, 1):
        lines.append(
            f"{i}. {r.get('full_name', '?')}  ⭐{r.get('stargazers_count', 0)}  "
            f"语言:{r.get('language') or '未知'}  更新:{r.get('updated_at', '?')[:10]}\n"
            f"   {r.get('description') or '（无简介）'}"
        )
    return "\n".join(lines)


def search_arxiv_papers(query: str, top_k: int = 5) -> str:
    """按关键字搜索 arXiv 学术论文，返回标题、摘要、作者、链接。

    适合：AI 学术前沿（"RAG 最近有什么新论文"）、技术原理论证。

    Args:
        query: 搜索关键字，如 "retrieval augmented generation"
        top_k: 返回条数，默认 5，最大 10
    """
    import xml.etree.ElementTree as ET

    try:
        resp = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": min(top_k, 10)},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        # arXiv API 返回 Atom XML，解析 entry 节点
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("a:entry", ns)
    except httpx.TimeoutException:
        print("[MCP][arXiv] 请求超时")
        return "arXiv 搜索超时，请稍后重试或改用网页搜索。"
    except Exception as e:
        print(f"[MCP][arXiv] 搜索失败：{e}")
        return f"arXiv 搜索失败：{type(e).__name__}: {e}"

    if not entries:
        return "arXiv 上没有找到相关论文。"

    lines = [f"arXiv 论文搜索结果："]
    for i, e in enumerate(entries, 1):
        title = (e.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
        summary = (e.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
        link = e.findtext("a:id", "", ns)
        lines.append(
            f"{i}. {title}\n"
            f"   作者: {', '.join(authors[:3])}{' 等' if len(authors) > 3 else ''}\n"
            f"   摘要: {summary[:150]}{'...' if len(summary) > 150 else ''}\n"
            f"   链接: {link}"
        )
    return "\n".join(lines)


def search_hf_models(query: str, top_k: int = 5) -> str:
    """按关键字搜索 HuggingFace 模型，返回模型名、任务类型、下载量、点赞数。

    适合：模型选型（"中文 embedding 模型推荐"、"哪个小模型适合本地跑"）。

    Args:
        query: 搜索关键字，如 "bge embedding chinese"
        top_k: 返回条数，默认 5，最大 10
    """
    try:
        resp = httpx.get(
            "https://huggingface.co/api/models",
            params={"search": query, "limit": min(top_k, 10), "sort": "downloads", "direction": -1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        models = resp.json()

        # 鲁棒兜底：HF 对多词搜索很严格，整串查不到就用第一个词单搜
        # （例："bge embedding chinese" 会空，但 "bge" 有一堆）
        if not models and query.strip():
            first_word = query.strip().split()[0]
            resp2 = httpx.get(
                "https://huggingface.co/api/models",
                params={"search": first_word, "limit": min(top_k, 10), "sort": "downloads", "direction": -1},
                timeout=_TIMEOUT,
            )
            resp2.raise_for_status()
            models = resp2.json()
    except httpx.TimeoutException:
        print("[MCP][HF] 请求超时")
        return "HuggingFace 搜索超时，请稍后重试或改用网页搜索。"
    except Exception as e:
        print(f"[MCP][HF] 搜索失败：{e}")
        return f"HuggingFace 搜索失败：{type(e).__name__}: {e}"

    if not models:
        return "HuggingFace 上没有找到相关模型。"

    lines = [f"HuggingFace 模型搜索结果（按下载量排序）："]
    for i, m in enumerate(models, 1):
        downloads = m.get("downloads", 0)
        # 下载量超过万级的转成 万/亿 更好读
        if downloads >= 1_0000_0000:
            dl_str = f"{downloads/1_0000_0000:.1f}亿"
        elif downloads >= 1_0000:
            dl_str = f"{downloads/1_0000:.1f}万"
        else:
            dl_str = str(downloads)
        lines.append(
            f"{i}. {m.get('id', '?')}  ⬇{dl_str}  ❤{m.get('likes', 0)}\n"
            f"   任务: {m.get('pipeline_tag') or '通用'}  库: {m.get('library_name') or '未知'}"
        )
    return "\n".join(lines)
