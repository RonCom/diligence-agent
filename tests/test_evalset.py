"""The committed eval set: shape, coverage and split. Runs in CI without sources."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from diligence_agent import evalset
from diligence_agent.grading import check

PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
CHECKS = {"date", "yes_no", "trial_status", "nct_set", "money", "contains_all", "none_listed", "decline", "judge"}


@pytest.fixture(scope="module")
def qs():
    return evalset.load(PATH)


def test_shape(qs):
    assert 85 <= len(qs) <= 100
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["check"]["type"] in CHECKS, q["id"]
        assert q["split"] in ("dev", "test") and q["style"] in ("direct", "plain")
        assert q["reference"] and "?" in q["question"], q["id"]
        if q["answerable"] and q["check"]["type"] != "none_listed":
            assert q["sources"], f"{q['id']} has no source record"


def test_coverage(qs):
    cats = Counter(q["category"] for q in qs)
    assert cats["out_of_scope"] == 15
    assert {"patent", "exclusivity", "approval", "indication", "trial", "readouts", "tenk"} <= set(cats)
    assert sum(q["style"] == "plain" for q in qs) >= 20
    assert sum(not q["answerable"] for q in qs) == 15


def test_split_stratified(qs):
    for cat in {q["category"] for q in qs}:
        items = [q for q in qs if q["category"] == cat]
        dev = sum(q["split"] == "dev" for q in items)
        assert 1 <= dev < len(items) or len(items) == 1, cat


def test_reference_passes_its_own_check(qs):
    """Each reference answer, graded by the question's own check, scores 1: the checks and golds agree."""
    for q in qs:
        spec = q["check"]
        if spec["type"] in ("judge", "decline", "none_listed", "yes_no", "nct_set", "trial_status"):
            continue
        assert check(spec, q["reference"]) == 1.0, (q["id"], q["reference"], spec)


def test_split_assignment_is_deterministic():
    qs = [{"id": f"x{i:02d}", "category": "a" if i < 9 else "b"} for i in range(12)]
    evalset.assign_splits(qs)
    first = [q["split"] for q in qs]
    evalset.assign_splits(qs)
    assert [q["split"] for q in qs] == first
    assert sum(q["split"] == "dev" for q in qs if q["category"] == "a") == 3


def test_diff_lists_changed_gold():
    old = [{"id": "a", "gold": {"x": 1}, "check": {}}, {"id": "b", "gold": {}, "check": {}}]
    new = [{"id": "a", "gold": {"x": 2}, "check": {}}, {"id": "c", "gold": {}, "check": {}}]
    assert evalset.diff(old, new) == ["a: gold {'x': 1} -> {'x': 2}", "c: new", "b: removed"]
