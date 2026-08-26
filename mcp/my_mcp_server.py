from mcp.server import FastMCP
import datetime
mcp = FastMCP("MyTools") #第五步

@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b

@mcp.tool()
def get_time() -> str:
    """返回当前服务器时间。"""

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    mcp.run()   # 默认 stdio 传输，等 Client 来连