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

级别也可用环境变量覆盖（参数优先级更高）：
    LOG_LEVEL=DEBUG python main.py

注意：必须在 import 业务模块（multi_agent.*、streamlit_1.*）之前调用，
因为 config.py 等模块在 import 时就会打日志，先配 handler 才不丢。
"""
import logging
import os
import sys

_configured = False  # 幂等：入口之间互相 import 时，重复调用不重复挂 handler


def setup_logging(level=None, log_file=None):
    """配置一次根日志器。重复调用安全（第二次起直接返回）。"""
    global _configured
    if _configured:
        return

    # 级别优先级：显式参数 > LOG_LEVEL 环境变量 > INFO
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        numeric = level
    else:
        numeric = logging.getLevelName(str(level).upper())
        if not isinstance(numeric, int):     # 写错级别名（如 LOG_LEVEL=verbose）
            numeric = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric)

    # MCP server 子进程可能已用 logging.basicConfig 配好 handler（也是 stderr），
    # root 已有 handler 就不再重复挂，只统一级别。
    if not root.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stderr)   # 一律 stderr，绝不 stdout
        handler.setFormatter(fmt)
        handler.setLevel(numeric)
        root.addHandler(handler)

        if log_file:                               # 本地脚本想留档时可选，默认不写文件
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)

    _configured = True
