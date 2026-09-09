"""
统一日志配置 —— 所有模块共用一个出口（logging_setup）。

为什么需要它：
  每个模块自己 print() 的最大问题是没有「级别」和「归属」——
  分不清是流程日志、警告还是错误；看不出是哪一行、哪个模块打的；
  更没法「平时安静、排查时打开」。stdlib logging 是企业基线，
  langchain / uvicorn / httpx 等框架本来就走 logging，配置好后它们的
  日志也会一起进这套格式，排查问题时不用东一个 print 西一个 print。

这套配置解决三件事：
  1. 级别：INFO(默认，看流程主干) / DEBUG(排查细节) / WARNING / ERROR
  2. 归属：每条日志带模块名，一眼认出是谁打的（multi_agent.hyde 等）
  3. 出口统一到 stderr：
     - 容器/服务场景日志本就走 stdout/stderr，由运行时收集，不自己写文件
     - ★ MCP Server 是 stdio 子进程，它的 stdout 是 JSON-RPC 协议通道，
       任何跑到 stdout 的日志都会污染协议流 —— 所以日志一律走 stderr

用法（每个进程入口调一次；入口 = main.py / backend.py / 评估脚本 / MCP server）：
    from multi_agent.logging_setup import setup_logging
    setup_logging()            # 默认 INFO
    setup_logging("DEBUG")     # 想看检索内部细节
    setup_logging("WARNING")   # 评估脚本：只要警告和错误

级别也可用环境变量覆盖（参数优先级更高，数字字符串也认）：
    LOG_LEVEL=DEBUG python main.py

注意：
  - 必须在 import 业务模块（multi_agent.*、streamlit_1.*）之前调用，
    因为 config.py 等模块在 import 时就会打日志，先配 handler 才不丢。
  - 进程内第一个调用生效（first-call-wins），后续调用直接返回；
    想排障时临时提级，重启进程并设 LOG_LEVEL=DEBUG 即可。
  - 锁只保证「本进程内多线程」不重复挂 handler；logging 是进程级的，
    跨进程各配各的，不存在也不需要一个全局跨进程锁。
"""
import logging
import os
import sys
import threading

_configured = False              # 进程内是否已挂过 handler
_lock = threading.Lock()         # 线程安全：并发调用也只配置一次


def _resolve_level(level):
    """把 显式参数 / LOG_LEVEL 环境变量 / 数字字符串 归一成 logging 级别 int。"""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    s = str(level).strip().upper()
    try:
        return int(s)            # "15" / "20" 这类数字字符串也认
    except ValueError:
        pass
    numeric = logging.getLevelName(s)      # "INFO"/"DEBUG"/"WARN" → int
    if not isinstance(numeric, int):       # 级别名写错（如 verbose）→ 兜底 INFO
        return logging.INFO
    return numeric


def setup_logging(level=None, log_file=None):
    """进程入口调用一次：定死级别、把日志出口接到 stderr。

    幂等 + 线程安全：handler 只挂一次；级别以第一个调用为准。
    """
    global _configured
    with _lock:
        if _configured:
            return

        numeric = _resolve_level(level)

        root = logging.getLogger()
        root.setLevel(numeric)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%m-%d %H:%M:%S",        # 服务日志跨天，带日期才好定位
        )
        # 判重用 type 精确匹配：FileHandler 也是 StreamHandler 的子类，
        # 若用 isinstance 判断，已挂文件 handler 时会误判"有控制台 handler"
        # 而漏挂 stderr，导致控制台静音。
        if not any(type(h) is logging.StreamHandler for h in root.handlers):
            sh = logging.StreamHandler(sys.stderr)   # 一律 stderr，绝不 stdout
            sh.setFormatter(fmt)
            root.addHandler(sh)

        if log_file and not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")   # 可选留档
            fh.setFormatter(fmt)
            root.addHandler(fh)

        _configured = True
