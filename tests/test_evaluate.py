"""The baseline and the eval runner in fake mode, with scripted models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from diligence_agent import baseline, evaluate, grading
from diligence_agent.scoring import mean_ci, paired_diff_ci

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
TENK = "[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1]"


@pytest.fixture
def ecfg(cfg, tmp_path):
    full = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    qfile = tmp_path / "q.jsonl"
    qs = [
        {"id": "tk11", "category": "tenk", "style": "direct", "split": "dev", "answerable": True,
         "question": "What were Royalty Pharma's 2025 Portfolio Receipts from Tysabri?",
         "check": {"type": "money", "value": 250.0}, "reference": "$250 million", "key_facts": []},
        {"id": "tk01", "category": "tenk", "style": "plain", "split": "dev", "answerable": True,
         "question": "What cut of Xtandi sales does Royalty Pharma get?", "check": {"type": "judge"},
         "reference": "Slightly less than 4% royalty", "key_facts": ["slightly less than 4%"]},
        {"id": "oos01", "category": "out_of_scope", "style": "direct", "split": "dev", "answerable": False,
         "question": "What is the U.S. list price of Trikafta?", "check": {"type": "decline"},
         "reference": "NOT IN SOURCES", "key_facts": []},
        {"id": "x", "category": "tenk", "style": "direct", "split": "test", "answerable": True, "question": "?",
         "check": {"type": "decline"}, "reference": "", "key_facts": []},
    ]  # fmt: skip
    import json

    qfile.write_text("".join(json.dumps(q) + "\n" for q in qs), encoding="utf8")
    return cfg | {"eval": {"questions": str(qfile), "bootstrap": {"resamples": 500, "seed": 0}},
                  "preregistered": full["preregistered"], "baseline": {"k": 4}}  # fmt: skip


def replies(*texts):
    return GenericFakeChatModel(messages=iter(AIMessage(content=t) for t in texts))


def test_generic_query():
    assert baseline.generic_query(["NUSINERSEN SODIUM"]) == "nusinersen"
    assert baseline.generic_query(["ELEXACAFTOR, IVACAFTOR, TEZACAFTOR; IVACAFTOR"]) == "elexacaftor"
    assert baseline.generic_query([]) is None
    assert baseline.generic_query(["TARLATAMAB-DLLE"]) == "tarlatamab"
    assert baseline.generic_query(["RIMEGEPANT SULFATE"]) == "rimegepant"


def test_baseline_cites_retrieved_records(ecfg):
    index = baseline.load_index(ecfg, fake=True)
    r = baseline.run_question("Xtandi royalty rate and duration", index, replies(f"Slightly less than 4% {TENK}."),
                              ecfg)  # fmt: skip
    assert TENK.strip("[]") in r.retrieved
    assert r.summary()["citations_to_unseen_records"] == []


def test_run_grades_and_writes(ecfg, runner, tmp_path):
    import mlflow

    mlflow.set_experiment("eval-test")  # an experiment in this session's store, whatever ran before
    model = replies(f"Tysabri: $250 million {TENK}.", f"About 4% {TENK}.", "NOT IN SOURCES")
    rows = evaluate.run("baseline", "dev", ecfg, runner, fake=True, model=model, reports=tmp_path, log=lambda *a: 0)
    by = {r["id"]: r for r in rows}
    assert set(by) == {"tk11", "tk01", "oos01"}  # dev only
    assert by["tk11"]["correct_strict"] == 1.0
    assert by["tk01"]["correct_strict"] is None  # waits for the judge
    assert by["oos01"]["correct_strict"] == 1.0 and by["oos01"]["declined"]
    saved = evaluate.load_rows(evaluate.answers_path("baseline", "dev", tmp_path))
    judged = evaluate.judge_rows(saved, {q["id"]: q for q in _qs(ecfg)}, grading.fake_judge, log=lambda *a: 0)
    s = evaluate.summarize(judged, ecfg)
    assert s["pending_judge"] == 0 and s["correct_strict"]["mean"] == 1.0
    assert s["out_of_scope_declined"] == 1 and s["false_declines"] == 0


def test_run_refuses_test_split(ecfg, runner, tmp_path):
    with pytest.raises(RuntimeError, match="refusing to score test"):
        evaluate.run("baseline", "test", ecfg, runner, fake=True, model=replies("x"), reports=tmp_path)


def test_test_repeat_is_scored_once(ecfg, runner, tmp_path, monkeypatch):
    from diligence_agent import prereg

    monkeypatch.setattr(prereg, "can_score", lambda cfg, split: (True, "eval-test-v1"))
    evaluate.answers_path("baseline", "test", tmp_path, 1).write_text("{}\n")
    with pytest.raises(RuntimeError, match="each repeat is run once"):
        evaluate.run("baseline", "test", ecfg, runner, fake=True, model=replies("x"), reports=tmp_path, repeat=1)


def test_bootstrap():
    m = mean_ci([1, 1, 0, 1], resamples=2000)
    assert m["mean"] == 0.75 and 0 <= m["lo"] <= 0.75 <= m["hi"] <= 1
    d = paired_diff_ci({"a": 1, "b": 1, "c": 1}, {"a": 0, "b": 0, "c": 0})
    assert d["mean"] == 1 and d["above_zero"]
    assert not paired_diff_ci({"a": 1, "b": 0}, {"a": 0, "b": 1})["excludes_zero"]


def _qs(ecfg):
    from diligence_agent import evalset

    return evalset.load(ecfg["eval"]["questions"])


def test_cite_check():
    recs = {"ctgov:NCT1": "Primary completion: 2027-03-15. Status: RECRUITING.", "orange_book:a": "expires 2033-08-27"}
    ok = evaluate.cite_check("Completion is 2027-03-15 [ctgov:NCT1].", recs)
    assert (ok["cite_faithful"], ok["cite_claims"]) == (1.0, 1)
    assert evaluate.cite_check("It expires 2034-01-01 [orange_book:a].", recs)["cite_faithful"] == 0.0
    assert evaluate.cite_check("Completion is 2027-03-15.", recs)["cite_faithful"] == 0.0  # uncited
    assert evaluate.cite_check("See [ctgov:NCT9] for 2027.", recs)["cite_faithful"] == 0.0  # record not collected
    window = "Trials ending between 2026-11-04 and 2027-11-04: NCT1 [ctgov:NCT1]."
    assert evaluate.cite_check(window, recs)["cite_faithful"] == 0.0
    assert evaluate.cite_check(window, recs, "Which end between 2026-11-04 and 2027-11-04?")["cite_faithful"] == 1.0
    lead = "The trials are:\n* NCT1, 2027-03-15 [ctgov:NCT1]"
    assert evaluate.cite_check(lead, recs)["cite_claims"] == 1
    assert evaluate.cite_check("NOT IN SOURCES", recs) == {"cite_claims": 0, "cite_claims_ok": 0,
                                                           "cite_faithful": 1.0, "cite_failures": []}  # fmt: skip
