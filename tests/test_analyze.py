from __future__ import annotations

import pytest

from diligence_agent import analyze, evaluate

TESTS = [
    {"id": "H1", "claim": "a", "compare": ["agent", "baseline"], "pass_if": "CI above 0"},
    {"id": "H3", "claim": "c", "compare": ["agent", "langchain"], "pass_if": "CI excludes 0"},
    {"id": "H4", "claim": "d", "pass_if": "at least 9 of 10", "applies_to": ["agent", "langchain"]},
]
CFG = {"preregistered": {"status": "frozen", "version": 1, "test_tag": "t", "tests": TESTS},
       "eval": {"bootstrap": {"resamples": 500, "seed": 0}}}  # fmt: skip


def rows(scores: list[float | None], declines: int = 0, oos: int = 0, oos_declined: int = 0, tag: str = "") -> list:
    out = []
    for i, s in enumerate(scores):
        out.append({"id": f"q{i}", "answer": f"a{i}{tag}", "answerable": True, "check_type": "date",
                    "declined": i < declines, "correct_strict": s, "correct_raw": s, "category": "c",
                    "style": "direct", "citations_to_unseen_records": [], "citations": 1, "tool_call_validity": 1,
                    "model_calls": 1, "budget_stop": False, "repeated_calls": 0, "latency_s": 1.0,
                    "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0})  # fmt: skip
    for j in range(oos):
        out.append(out[0] | {"id": f"o{j}", "answerable": False, "check_type": "decline",
                             "declined": j < oos_declined, "correct_strict": float(j < oos_declined)})  # fmt: skip
    return out


def test_answers_path_per_repeat(tmp_path):
    assert evaluate.answers_path("agent", "dev", tmp_path).name == "answers_agent_dev.jsonl"
    assert evaluate.answers_path("agent", "test", tmp_path, 2).name == "answers_agent_test_r2.jsonl"


def test_tests_and_rerun(tmp_path):
    good, bad = [1.0] * 20, [1.0] * 10 + [0.0] * 10
    for s, sc, decl, od in (("agent", good, 0, 10), ("baseline", bad, 2, 10), ("langchain", good, 0, 8)):
        evaluate.write_rows(rows(sc, decl, oos=10, oos_declined=od), evaluate.answers_path(s, "test", tmp_path, 1))
    evaluate.write_rows(
        rows(good, oos=10, oos_declined=10, tag="x"), evaluate.answers_path("agent", "test", tmp_path, 2)
    )
    res = analyze.analyze(CFG, "test", tmp_path)
    by = {t["id"]: t for t in res["tests"]}
    assert by["H1"]["result"] == "pass" and by["H1"]["false_declines"] == {"agent": 0, "baseline": 2}
    assert by["H3"]["result"] == "no difference shown"  # langchain loses 2 out-of-scope points: CI covers 0
    assert by["H4"]["per_system"]["agent"]["passed"] and not by["H4"]["per_system"]["langchain"]["passed"]
    assert res["rerun"]["agent"]["answers_changed"] == 30 and res["rerun"]["agent"]["grade_changed"] == 0
    assert "baseline" not in res["rerun"]


def test_refuses_pending_grades(tmp_path):
    evaluate.write_rows(rows([1.0, None]), evaluate.answers_path("agent", "test", tmp_path, 1))
    with pytest.raises(RuntimeError, match="pending"):
        analyze.analyze(CFG, "test", tmp_path)
