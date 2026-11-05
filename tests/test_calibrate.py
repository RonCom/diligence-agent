from __future__ import annotations

import csv

import pytest

from diligence_agent import calibrate as cal


def row(system, qid, cat, answer, judge_correct=None, judge_faithful=None, check="date", correct_raw=1.0):
    return {"system": system, "id": qid, "category": cat, "question": f"{qid}?", "answer": answer,
            "records": {"orange_book:x": "rec text"}, "judge_correct": judge_correct,
            "judge_faithful": judge_faithful, "check_type": check, "correct_raw": correct_raw}  # fmt: skip


QS = {"tk01": {"check": {"type": "judge"}, "reference": "4%"}, "pat01": {"check": {"type": "date"}, "reference": "d"},
      "oos01": {"check": {"type": "decline"}, "reference": "NOT IN SOURCES"},
      "app01": {"check": {"type": "date"}, "reference": "d"}}  # fmt: skip


def test_kappa():
    assert cal.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cal.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert cal.cohen_kappa([1, 1, 1], [1, 1, 1]) is None  # undefined with one shared label
    assert cal.cohen_kappa([True, True, False, True], [True, False, False, True]) == pytest.approx(0.5)


def test_sample_takes_all_judged_then_spreads_categories():
    rows = [row(s, "tk01", "tenk", "a", check="judge") for s in ("agent", "baseline")]
    rows += [
        row("agent", q, c, "a") for q, c in (("pat01", "patent"), ("oos01", "out_of_scope"), ("app01", "approval"))
    ]
    picked = cal.sample(rows, QS, n=4)
    assert sum(r["id"] == "tk01" for r in picked) == 2
    assert len({r["category"] for r in picked}) == 3


def test_sheet_is_blind_and_calibration_matches_exact_answers(tmp_path):
    rows = [row("agent", "tk01", "tenk", "About 4% [orange_book:x].", 1, 1, check="judge"),
            row("agent", "pat01", "patent", "2037 [orange_book:x].", 1, 1),
            row("baseline", "pat01", "patent", "2030.", 0, 0, correct_raw=0.0),
            row("baseline", "app01", "approval", "2012.", 1, 0)]  # fmt: skip
    csv_path, guide = tmp_path / "labels.csv", tmp_path / "LABELING.md"
    cal.write_sheet(rows, QS, csv_path, guide)
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "judge_correct" not in text and "rec text" in text
    assert "baseline" not in text and "agent" not in text.replace("Answer", "")  # blind to the system
    key = __import__("json").loads(cal.key_path(csv_path).read_text())
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    grades = {("agent", "tk01"): ("1", "1"), ("agent", "pat01"): ("1", "1"), ("baseline", "pat01"): ("0", "1"),
              ("baseline", "app01"): ("1", "0")}  # fmt: skip
    for lab in labels:
        lab["human_correct"], lab["human_faithful"] = grades[(key[lab["label_id"]]["system"], lab["question_id"])]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cal.FIELDS)
        w.writeheader()
        w.writerows(labels)
    with pytest.raises(RuntimeError, match="already has labels"):
        cal.write_sheet(rows, QS, csv_path, guide)

    rows[3]["answer"] = "changed on a rerun"  # this label no longer counts
    out = cal.calibrate(csv_path, {"agent": rows[:2], "baseline": rows[2:]})
    assert out["matched"] == 3 and len(out["stale_labels"]) == 1
    assert out["correctness"]["agreement"] == 1.0 and out["correctness"]["cohen_kappa"] == 1.0
    assert out["faithfulness"]["n"] == 3 and out["faithfulness"]["agreement"] == pytest.approx(2 / 3)
    assert out["judge_vs_code"]["n"] == 3


def test_hand_labels_and_fallback(tmp_path):
    import json

    assert cal.key_path(tmp_path / "human_labels.csv").name == "label_key.json"
    assert cal.key_path(tmp_path / "human_labels_test.csv").name == "human_labels_test_key.json"
    rows = [row("agent", "tk01", "tenk", "About 4%.", check="judge"), row("baseline", "tk01", "tenk", "About 4%.",
            check="judge"), row("agent", "pat01", "patent", "2037.")]  # fmt: skip
    picked = cal.judged_unlabeled(rows, QS, set())
    assert len(picked) == 1  # same answer to the same question: one label covers both
    sheet = tmp_path / "human_labels_test.csv"
    cal.write_sheet(picked, QS, sheet, tmp_path / "LABELING_test.md", guide=cal.LABEL_GUIDE)
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    labels[0]["human_correct"] = "0.5"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cal.FIELDS)
        w.writeheader()
        w.writerows(labels)
    got = cal.hand_labels(tmp_path)
    assert got == {("tk01", cal.answer_sha("About 4%.")): 0.5}
    assert cal.judged_unlabeled(rows, QS, set(got)) == []
    assert not cal.labels_replace_judge(tmp_path)  # no calibration yet
    (tmp_path / "judge_calibration.json").write_text(json.dumps({"judge_counts": False}))
    assert cal.labels_replace_judge(tmp_path)
