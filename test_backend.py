"""
测试 FastAPI 后端完整链路：
  1. 上传 PDF → 拿到 session_id
  2. 流式对话 → 打印 SSE 事件和最终答案
"""
import requests
import json

BASE = "http://127.0.0.1:8000"

# ========== 步骤 1：上传 PDF ==========
print("=" * 50)
print("步骤 1：上传 PDF")
print("=" * 50)

# 用你 docs 目录下的 PDF 做测试
import glob
pdf_files = glob.glob("D:/python2/lc_course/docs/*.pdf")
if not pdf_files:
    # 备选：找 outputs 目录
    pdf_files = glob.glob("D:/claude/Claude_Outputs/*.pdf")

print(f"找到 {len(pdf_files)} 个 PDF 文件")
if not pdf_files:
    print("❌ 没找到 PDF，请确认路径")
    exit(1)

# 构建上传请求
files = []
for path in pdf_files[:3]:  # 只传前 3 个，省时间
    files.append(("files", (path.split("\\")[-1], open(path, "rb"), "application/pdf")))

resp = requests.post(f"{BASE}/api/upload-pdf", files=files)

# 关闭文件句柄
for _, (_, fh, _) in files:
    fh.close()

if resp.status_code != 200:
    print(f"❌ 上传失败: {resp.status_code} {resp.text}")
    exit(1)

data = resp.json()
session_id = data["session_id"]
print(f"✅ 上传成功！session_id = {session_id}")
print(f"   页数: {data['page_count']}, child chunks: {data['child_chunks']}, parent chunks: {data['parent_chunks']}")


# ========== 步骤 2：流式对话 ==========
print()
print("=" * 50)
print("步骤 2：流式对话")
print("=" * 50)

question = "什么是RAG？"
print(f"问题: {question}")
print()

resp = requests.post(
    f"{BASE}/api/chat-stream",
    json={"session_id": session_id, "question": question, "max_tool_rounds": 5},
    stream=True,
)

if resp.status_code != 200:
    print(f"❌ 对话失败: {resp.status_code} {resp.text}")
    exit(1)

# 逐行读取 SSE 事件流
for line in resp.iter_lines():
    if not line:
        continue
    line = line.decode()
    if not line.startswith("data: "):
        continue
    data = json.loads(line[6:])  # 去掉 "data: " 前缀

    event_type = data.get("type")
    if event_type == "status":
        agent = data.get("agent")
        if agent == "researcher":
            print(f"  🔍 Researcher 第 {data.get('rounds', '?')} 轮搜索中...")
        elif agent == "writer":
            print(f"  ✍️ Writer 正在撰写答案...")
    elif event_type == "tool_call":
        print(f"    🔧 {data['name']}({data['args']})")
    elif event_type == "done":
        print()
        print("=" * 50)
        print("最终回答：")
        print("=" * 50)
        print(data["answer"])
        break

print()
print("✅ 测试完成")
