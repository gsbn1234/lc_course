"""
pytest 全局夹具：把项目根目录加进 sys.path，
让测试文件里能 `from multi_agent.xxx import ...`。
（pytest 默认只把"用例所在目录"加进 import 路径，不会自动认识 multi_agent 包。）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
