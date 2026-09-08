import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ========== LangSmith 全链路追踪开关 ==========
# 原理：LangChain/LangGraph 每次 run 开始时自动读环境变量，决定是否把轨迹上报到 LangSmith。
# 只要在进程启动早期把变量设好即可，无需改任何业务代码 —— config.py 是几乎所有模块的公共依赖，
# 在这里设置等于对 main.py / backend.py / 评估脚本一次性全局生效。
#
# 启用条件（二者缺一就静默关闭，不报错）：
#   1. LANGSMITH_TRACING=true
#   2. 填了真实的 LANGSMITH_API_KEY（https://smith.langchain.com → Settings → API Keys，lsv2_ 开头）
LANGSMITH_TRACING = (
    os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
    and bool(os.getenv("LANGSMITH_API_KEY"))
)
if LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "doc-intel")
    print("[LangSmith] ✅ 全链路追踪已开启 → https://smith.langchain.com")
else:
    # 防呆：如果 .env 里开了开关却没填 Key，LangSmith 会一路告警刷屏，这里强制关掉。
    # 没填 Key = 本地自测，静默跳过，功能完全不受影响。
    os.environ["LANGSMITH_TRACING"] = "false"

# 项目根目录：以本文件（multi_agent/config.py）为基准向上两级。
# 无论从哪个目录启动脚本，路径都稳定指向项目根，不再依赖 CWD。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 知识库原始文档目录（docs/ 下的 PDF）
PDF_PATH = str(PROJECT_ROOT / "docs")

# FAISS 向量库索引目录
FAISS_PATH = str(PROJECT_ROOT / "faiss_db")

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)