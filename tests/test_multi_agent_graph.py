"""多 Agent 图测试：核心是 Reviewer 节点的行为 + 图的接线是否正确。全程不联网。"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from multi_agent.multi_agent_graph import (
    build_multi_agent_graph,
    extract_answer,
    reviewer_agent_node,
    route_researcher,
    route_reviewer,
    route_supervisor,
)


class FakeLLM:
    """不联网的假 LLM：bind_tools 原样返回，invoke 返回预设内容。"""

    def __init__(self, content="OK"):
        self._content = content

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content=self._content)


# ---------- 图结构 ----------

def test_graph_contains_all_nodes():
    graph = build_multi_agent_graph(FakeLLM(), None, None, None, None)
    nodes = set(graph.get_graph().nodes)
    assert {"rewrite_query_node", "supervisor_node", "researcher_agent",
            "researcher_tool_node", "writer_agent", "reviewer_agent"} <= nodes


def test_graph_writer_goes_through_reviewer():
    """Writer 写完必须经过 Reviewer，而不是直接结束。"""
    graph = build_multi_agent_graph(FakeLLM(), None, None, None, None)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("writer_agent", "reviewer_agent") in edges


def test_graph_reviewer_branches_to_researcher_and_end():
    graph = build_multi_agent_graph(FakeLLM(), None, None, None, None)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("reviewer_agent", "researcher_agent") in edges  # REVISE 打回
    assert ("reviewer_agent", END) in edges                 # PASS 结束


# ---------- 路由 ----------

def test_route_reviewer_pass_ends():
    assert route_reviewer({"next_agent": "finish"}) == END


def test_route_reviewer_revise_back_to_researcher():
    assert route_reviewer({"next_agent": "researcher"}) == "researcher_agent"


def test_route_researcher_continue_search():
    assert route_researcher({"next_agent": "researcher"}) == "researcher_tool_node"


def test_route_researcher_done_to_supervisor():
    assert route_researcher({"next_agent": "writer"}) == "supervisor_node"


def test_route_supervisor_sends_writer():
    assert route_supervisor({"next_agent": "writer"}) == "writer_agent"


# ---------- Reviewer 节点行为 ----------

def _state(writer_answer, rounds=0, max_rounds=2, research="研究材料", **extra):
    return {
        "question": "测试问题",
        "research_results": research,
        "writer_messages": [AIMessage(content=writer_answer)],
        "review_rounds": rounds,
        "max_review_rounds": max_rounds,
        **extra,
    }


def test_reviewer_pass_ends():
    llm = FakeLLM("VERDICT: PASS\nFEEDBACK: 无需修改")
    out = reviewer_agent_node(_state("合格答案"), llm)
    assert out["review_verdict"] == "PASS"
    assert out["next_agent"] == "finish"
    assert out["review_rounds"] == 1


def test_reviewer_revise_pushes_feedback_to_researcher():
    llm = FakeLLM("VERDICT: REVISE\nFEEDBACK: 缺少数据来源，请补充搜索")
    out = reviewer_agent_node(_state("不完整答案"), llm)
    assert out["review_verdict"] == "REVISE"
    assert out["next_agent"] == "researcher"
    assert out["review_rounds"] == 1
    # 修改意见必须变成 HumanMessage 进 researcher_messages，Researcher 才能"带着意见重搜"
    feedbacks = [m for m in out.get("researcher_messages", [])
                 if isinstance(m, HumanMessage) and "缺少数据来源" in m.content]
    assert feedbacks, "修改意见没有传给 Researcher"


def test_reviewer_revise_over_cap_forces_finish():
    """已达上限还 REVISE：强制放行，防死循环。"""
    llm = FakeLLM("VERDICT: REVISE\nFEEDBACK: 还是不行")
    out = reviewer_agent_node(_state("答案", rounds=2, max_rounds=2), llm)
    assert out["next_agent"] == "finish"


def test_reviewer_garbage_output_defaults_revise():
    """LLM 输出不符合 VERDICT/FEEDBACK 格式：宁严勿松，默认 REVISE。"""
    llm = FakeLLM("完全看不懂的输出")
    out = reviewer_agent_node(_state("答案"), llm)
    assert out["review_verdict"] == "REVISE"
    assert out["next_agent"] == "researcher"


def test_reviewer_no_answer_skips_review():
    """拿不到 Writer 的回答：直接放行，不产生死循环。"""
    out = reviewer_agent_node(_state(""), FakeLLM("VERDICT: REVISE\nFEEDBACK: x"))
    assert out["next_agent"] == "finish"


# ---------- 提取答案 ----------

def test_extract_answer():
    result = {"writer_messages": [AIMessage(content="最终答案")]}
    assert extract_answer(result) == "最终答案"


def test_extract_answer_fallback():
    assert extract_answer({}) == "（未能提取到最终回答）"
