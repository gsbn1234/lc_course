"""config.py 是全局路径和环境变量入口，先把它钉死。"""

from multi_agent import config


def test_project_root_points_to_project():
    """PROJECT_ROOT 应指向项目根（目录里躺着 main.py 的地方）。"""
    assert (config.PROJECT_ROOT / "main.py").exists()


def test_paths_resolve_from_root():
    """docs / faiss_db 路径应统一以项目根为基准，而不是写死的绝对路径。"""
    assert config.PDF_PATH.endswith("docs")
    assert config.FAISS_PATH.endswith("faiss_db")


def test_deepseek_key_loaded():
    """.env 能被读到：没有 key 项目根本跑不起来，这个断言是启动前的体检。"""
    assert config.DEEPSEEK_API_KEY, ".env 里没读到 DEEPSEEK_API_KEY"
