from __future__ import annotations

import json

import pytest

from diligence_agent import grading
from diligence_agent.grading import amounts_millions, check, dates_in, declined, grade, months_in


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ("NOT IN SOURCES", "exact"),
        ("not in sources.", "exact"),
        ("The Orange Book search for Tremfya did not return any records.", ""),  # states a finding, not a decline
        ("The records do not provide a list price for Trikafta.", "prose"),
        ("I cannot determine Vertex's gross margin from these sources.", "prose"),
        ("There is no information about Keytruda sales in the records.", "prose"),
        ("None of the listed trials name Xtandi as the intervention.", "prose"),
        ("The provided SEC filing from Royalty Pharma plc does not contain Keytruda sales.", "prose"),
        (
            "Patent 12161628 expires 2037-02-23 [orange_book:N203415/patent/12161628]. The sources do not list a PTE.",
            "",
        ),
    ],
)
def test_declined(answer, kind):
    assert declined(answer) == kind


def test_dates_and_months():
    text = "Expires February 23, 2037; another on 2033-09-11 and 9 Aug 2030, or 04/01/2029."
    assert dates_in(text) == ["2037-02-23", "2033-09-11", "2030-08-09", "2029-04-01"]
    assert {"2027-04", "2026-12"} <= months_in("Primary completion April 2027 (est.), earlier 2026-12.")


def test_markdown_escapes_ignored():
    spec = {"type": "trial_status", "status": "ACTIVE_NOT_RECRUITING", "primary_completion": "2027-03-15"}
    assert check(spec, r"NCT04455750 is ACTIVE\_NOT\_RECRUITING; primary completion 2027-03-15.") == 1.0
    assert check({"type": "money", "value": 170.0}, r"Receipts were \$170 million.") == 1.0
    assert grading.unescape(r"a\_b \$1 \\ c") == r"a_b $1 \ c"


def test_amounts():
    assert amounts_millions("$250 million") == [250.0]
    assert 6300.0 in amounts_millions("about $6.3 billion")
    assert 6297.0 in amounts_millions("6,297 million")


@pytest.mark.parametrize(
    ("spec", "answer", "score"),
    [
        ({"type": "date", "value": "2037-02-23"}, "Patent 12161628 expires February 23, 2037.", 1.0),
        ({"type": "date", "value": "2037-02-23"}, "One expires 2026-08-24; the last expires 2037-02-23.", 0.5),
        ({"type": "date", "value": "2037-02-23"}, "It expires 2026-08-24.", 0.0),
        ({"type": "yes_no", "value": True}, "Yes. Tremfya is approved for Crohn's disease [x].", 1.0),
        ({"type": "yes_no", "value": True}, "Yes, it is approved; there is no age limit in the label.", 1.0),
        ({"type": "yes_no", "value": False}, "No, Erleada is approved for nmCRPC, not nmCSPC.", 1.0),
        ({"type": "yes_no", "value": False}, "Tysabri is not approved for ulcerative colitis.", 1.0),
        ({"type": "yes_no", "value": False}, "Yes, it is indicated for that.", 0.0),
        ({"type": "yes_no", "value": False}, "NOT IN SOURCES", 0.0),  # a decline is not a "no"
        ({"type": "yes_no", "value": False}, "The records do not mention glioblastoma.", 0.0),
        (
            {"type": "trial_status", "status": "ACTIVE_NOT_RECRUITING", "primary_completion": "2027-07-05"},
            "NCT06211036 is active, not recruiting; primary completion July 5, 2027.",
            1.0,
        ),
        ({"type": "trial_status", "status": "RECRUITING", "primary_completion": "2027-10"}, "It is recruiting.", 0.5),
        ({"type": "nct_set", "value": ["NCT06211036", "NCT07778316"]}, "NCT06211036 and NCT07778316.", 1.0),
        ({"type": "nct_set", "value": ["NCT06211036", "NCT07778316"]}, "Only NCT06211036.", 0.5),
        ({"type": "nct_set", "value": ["NCT06211036", "NCT07778316"]}, "NCT01234567.", 0.0),
        ({"type": "money", "value": 250.0}, "Tysabri brought in $250 million in 2025.", 1.0),
        ({"type": "money", "value": 6297.0}, "Xtandi sold about $6.3 billion.", 1.0),
        ({"type": "money", "value": 250.0}, "About $1,665 million.", 0.0),
        ({"type": "contains_all", "value": ["2035", "2036"]}, "Through 2035-2036.", 1.0),
        ({"type": "contains_all", "value": ["September", "2026"]}, "Through 2026.", 0.0),
        ({"type": "none_listed", "value": None}, "No patents are listed; Tremfya is a biologic.", 1.0),
        ({"type": "none_listed", "value": None}, "NOT IN SOURCES", 1.0),
        ({"type": "none_listed", "value": None}, 'The Orange Book search for "Tremfya" returned no records.', 1.0),
        ({"type": "money", "value": 170.0}, "Receipts were **$170,357 thousand**.", 1.0),
        ({"type": "none_listed", "value": None}, "It expires in 2031.", 0.0),
        ({"type": "decline"}, "NOT IN SOURCES", 1.0),
        ({"type": "decline"}, "Trikafta costs about $26,000 a month.", 0.0),
    ],
)
def test_checks(spec, answer, score):
    assert check(spec, answer) == score


def test_decline_on_answerable_scores_zero_raw_kept():
    q = {"answerable": True, "check": {"type": "judge"}, "question": "q", "reference": "r", "key_facts": []}
    g = grade(q, "The records do not provide the royalty rate.", {"judge_correct": 1, "judge_faithful": 1})
    assert g["abstain_kind"] == "prose" and g["correct_raw"] == 1 and g["correct_strict"] == 0.0


def test_exact_decline_is_faithful_and_correct_out_of_scope():
    q = {"answerable": False, "check": {"type": "decline"}}
    g = grade(q, "NOT IN SOURCES", {"judge_faithful": 0})
    assert g["correct_strict"] == 1.0 and g["judge_faithful"] == 1.0 and g["judge_faithful_raw"] == 0


def test_biologic_decline_not_penalized():
    q = {"answerable": True, "check": {"type": "none_listed"}}
    assert grade(q, "NOT IN SOURCES")["correct_strict"] == 1.0


def test_judge_parses_and_falls_back():
    q = {"question": "q", "reference": "r", "key_facts": ["a"]}
    assert grading.judge(grading.fake_judge, q, "x", "")["judge_correct"] == 1

    def noisy(system, user):
        return "<think>hmm</think> Here: " + json.dumps({"correct": 0.5, "faithful": 1, "reason": "partial"})

    out = grading.judge(noisy, q, "x", "")
    assert out["judge_correct"] == 0.5 and out["judge_reason"] == "partial"


def test_hand_label_replaces_judge_on_judge_questions():
    q = {"check": {"type": "judge"}, "answerable": True}
    judged = {"judge_correct": 1, "judge_faithful": 1}
    assert grading.grade(q, "About 4% [x:y].", judged)["correct_source"] == "judge"
    g = grading.grade(q, "About 4% [x:y].", judged, human={"correct": 0.5})
    assert (g["correct_raw"], g["correct_strict"], g["correct_source"]) == (0.5, 0.5, "human")
    pending = grading.grade(q, "About 4% [x:y].", judged, human={"correct": None})
    assert pending["correct_strict"] is None  # unlabeled: pending, never the judge's grade
