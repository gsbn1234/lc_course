"""一键启动 LC Course Docker 项目。

用法（重启电脑后）：
  1. 在 PyCharm 里打开本项目，右键本文件 → Run；
  2. 或终端执行：python start_project.py
自动完成：启动 WSL Docker → 拉起容器 → 打开网页。
"""
import subprocess
import time
import webbrowser
import sys

WSL_DISTRO = "Ubuntu"
PROJECT_DIR = "/mnt/d/python2/lc_course"


def wsl(args: str, root: bool = False) -> bool:
    """在 WSL 里跑命令；root=True 免 sudo 密码启动 docker 服务。成功返回 True。"""
    cmd = ["wsl", "-d", WSL_DISTRO]
    if root:
        cmd += ["-u", "root"]
    cmd += ["-e", "bash", "-lc", args]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def main():
    print("[1/4] 启动 Docker 服务...")
    wsl("if ! docker info >/dev/null 2>&1; then service docker start; fi", root=True)

    print("[2/4] 等待 Docker 就绪...")
    for _ in range(60):          # 最多等 60 秒
        if wsl("docker info >/dev/null 2>&1"):
            break
        time.sleep(1)
    else:
        print("Docker 60 秒内未就绪，请检查 Docker 安装。")
        sys.exit(1)

    print("[3/4] 拉起项目容器...")
    if not wsl(f"cd {PROJECT_DIR} && docker compose up -d"):
        print("容器启动失败，请打开终端手动执行 docker compose up -d 看报错。")
        sys.exit(1)

    print("[4/4] 打开网页...")
    webbrowser.open("http://localhost:8501")
    print("完成！浏览器应已打开 http://localhost:8501")


if __name__ == "__main__":
    main()
