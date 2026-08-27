"""eval_judge 的数字解析容错测试（不联网，纯逻辑）。"""

from eval_judge import _parse_score


def test_parse_plain_number():
    assert _parse_score("0.75") == 0.75


def test_parse_number_with_punctuation():
    """LLM 偶尔输出带标点/解释：0.75。 / 0.8（不错）——必须能解析。"""
    assert _parse_score("0.75。") == 0.75
    assert _parse_score("0.8（不错）") == 0.8


def test_parse_garbage_falls_back():
    """完全解析不出数字：给 0.5 兜底，而不是让脚本崩掉。"""
    assert _parse_score("完全看不懂的输出") == 0.5


def test_parse_out_of_range_clamped():
    """超出 [0,1] 的异常分数被钳制。"""
    assert _parse_score("1.5") == 1.0
    assert _parse_score("-0.2") == 0.0
