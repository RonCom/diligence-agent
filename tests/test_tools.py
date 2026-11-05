from __future__ import annotations

import time

from pydantic import BaseModel

from diligence_agent.cache import ToolCache, cache_key
from diligence_agent.tools import build_runner
from diligence_agent.tools.base import Tool, ToolRunner
from diligence_agent.tools.ctgov import CtgovArgs


def by_id(outcome):
    return {e.record_id: e for e in outcome.evidence}


# ---- Orange Book --------------------------------------------------------------------------


def test_orange_book_collapses_rows(runner):
    out = runner.call("orange_book", {"drug": "Trikafta"})
    assert out.ok, out.error
    ids = by_id(out)
    assert {"N212273", "N217660"} <= set(ids)
    pat = ids["N212273/patent/9012496"]
    assert pat.fields["expires"] == "2033-07-15"
    assert "U-2649" in pat.text
    # 289 patent rows collapse to one record per (application, patent, expiry)
    patents = [e for e in out.evidence if "/patent/" in e.record_id]
    assert len(patents) == len({e.record_id for e in patents}) < 289


def test_orange_book_exclusivity_codes(runner):
    ids = by_id(runner.call("orange_book", {"drug": "TRIKAFTA"}))
    nce_like = ids["N212273/exclusivity/ODE-543/2033-03-27"]
    assert "orphan drug" in nce_like.text
    assert ids["N217660/exclusivity/NP/2026-04-26"].fields["expires"] == "2026-04-26"
    assert "ODE*" in {e.fields.get("code") for e in ids.values()}


def test_orange_book_matches_ingredient_and_misses(runner):
    assert runner.call("orange_book", {"drug": "ivacaftor"}).evidence
    miss = runner.call("orange_book", {"drug": "Keytruda"})
    assert miss.ok and miss.evidence == [] and miss.as_message() == "No records found."


# ---- ClinicalTrials.gov ---------------------------------------------------------------------


def test_ctgov_parses_studies(runner):
    out = runner.call("ctgov_search", {"intervention": "elexacaftor", "phase": "Phase 3"})
    assert out.ok and out.evidence
    s = by_id(out)["NCT03691779"]
    assert s.fields["phases"] == ["PHASE3"] and s.fields["status"] == "COMPLETED"
    assert s.fields["primary_completion"] == "2020-08-07"
    assert s.url == "https://clinicaltrials.gov/study/NCT03691779"


def test_ctgov_arg_repair():
    a = CtgovArgs.model_validate({"intervention": "xy", "phase": "phase 3", "status": "active, not recruiting"})
    assert a.phase == "PHASE3" and a.status == ["ACTIVE_NOT_RECRUITING"]
    assert CtgovArgs.model_validate({"intervention": "xy", "phase": 2}).phase == "PHASE2"


def test_ctgov_builds_filters(cfg):
    import httpx

    from diligence_agent.http import HttpClient
    from diligence_agent.tools.base import ToolContext
    from diligence_agent.tools.ctgov import CtgovTool

    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json={"studies": []})

    ctx = ToolContext(http=HttpClient(transport=httpx.MockTransport(handler)), data_dir=cfg["data_dir"])
    args = CtgovArgs(intervention="xy", phase="PHASE3", status=["RECRUITING"], primary_completion_from="2026-11-04")
    CtgovTool().run(args, ctx)
    assert seen["filter.overallStatus"] == "RECRUITING"
    assert seen["filter.advanced"] == "AREA[Phase]PHASE3 AND AREA[PrimaryCompletionDate]RANGE[2026-11-04,MAX]"


# ---- openFDA ----------------------------------------------------------------------------------


def test_fda_label_indications(runner):
    out = runner.call("fda_label", {"drug": "Trikafta"})
    assert out.ok and len(out.evidence) == 1
    e = out.evidence[0]
    assert "cystic fibrosis" in e.text.lower() and e.fields["effective"] == "2026-06-22"
    assert e.url.startswith("https://dailymed.nlm.nih.gov/")


def test_fda_label_no_match_is_empty(runner):
    out = runner.call("fda_label", {"drug": "Notadrug"})
    assert out.ok and out.evidence == []


def test_drugs_at_fda_original_and_efficacy(runner):
    ids = by_id(runner.call("drugs_at_fda", {"drug": "Trikafta"}))
    assert ids["NDA217660/ORIG-1"].fields["approved"] == "2023-04-26"
    assert ids["NDA217660/SUPPL-3"].fields["class"] == "EFFICACY"
    assert "NDA217660/SUPPL-9" not in ids  # labeling supplements are left out
    assert any(k.startswith("NDA212273/ORIG") for k in ids)


# ---- EDGAR ------------------------------------------------------------------------------------


def test_edgar_filings_by_ticker(runner):
    out = runner.call("edgar_filings", {"company": "exph", "form": "10-k", "limit": 1})
    assert out.ok and len(out.evidence) == 1
    e = out.evidence[0]
    assert e.fields["period"] == "2025-12-31"
    assert e.url == "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000007/exph-20251231.htm"


def test_edgar_requires_user_agent_live(cfg, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    live = build_runner(cfg, fake=False, use_cache=False)
    out = live.call("edgar_filings", {"company": "RPRX"})
    assert not out.ok and "SEC_USER_AGENT" in out.error


# ---- PubMed -----------------------------------------------------------------------------------


def test_pubmed_titles_and_abstracts(runner):
    out = runner.call("pubmed_search", {"query": "elexacaftor phase 3", "max_results": 2})
    assert out.ok and len(out.evidence) == 2
    e = out.evidence[0]
    assert e.record_id.isdigit() and e.url == f"https://pubmed.ncbi.nlm.nih.gov/{e.record_id}/"
    assert e.fields["title"] and "Abstract:" in e.text


def test_pubmed_no_hits(runner):
    assert runner.call("pubmed_search", {"query": "zzz unknown"}).evidence == []


# ---- Runner: validation, caching, timeouts ------------------------------------------------------


def test_validation_error_message_for_repair(runner):
    out = runner.call("ctgov_search", {"phase": "PHASE3"})
    assert not out.ok and out.error.startswith("invalid arguments: intervention")
    assert runner.call("ctgov_search", "{not json").error.startswith("arguments are not valid JSON")
    assert "available:" in runner.call("nope", {}).error


def test_schemas_have_no_titles(runner):
    schemas = runner.schemas()
    assert {s["function"]["name"] for s in schemas} == {
        "orange_book",
        "ctgov_search",
        "fda_label",
        "drugs_at_fda",
        "edgar_filings",
        "pubmed_search",
    }
    assert "title" not in str(schemas[0]["function"]["parameters"])


class _Args(BaseModel):
    q: str


class _Counting(Tool):
    name = "counting"
    description = "test"
    Args = _Args
    calls = 0

    def run(self, args, ctx):
        type(self).calls += 1
        from diligence_agent.evidence import Evidence

        return [Evidence(source="t", record_id=args.q, url="u", text="t")]


class _Slow(_Counting):
    name = "slow"

    def run(self, args, ctx):
        time.sleep(1.0)
        return []


def test_cache_hits_on_normalized_args(runner, tmp_path):
    r = ToolRunner([_Counting()], runner.ctx, cache=ToolCache(tmp_path / "c"))
    first = r.call("counting", {"q": "Trikafta  patents"})
    second = r.call("counting", {"q": "trikafta patents"})
    assert not first.cache_hit and second.cache_hit and _Counting.calls == 1
    assert first.key == second.key == cache_key("counting", {"q": "TRIKAFTA PATENTS"})
    assert second.evidence[0].record_id == "Trikafta  patents"


def test_tool_timeout(runner):
    r = ToolRunner([_Slow()], runner.ctx, timeout_s=0.1)
    out = r.call("slow", {"q": "x"})
    assert not out.ok and "timed out" in out.error


def test_orange_book_latest_expiry_summary(runner):
    out = runner.call("orange_book", {"drug": "Trikafta"})
    ids = by_id(out)
    patents = [e for e in out.evidence if "/patent/" in e.record_id and not e.fields["delisted"]]
    excl = [e for e in out.evidence if "/exclusivity/" in e.record_id]
    # drug-level record first, across both NDAs, computed from the patent records themselves
    top = out.evidence[0]
    assert top.record_id == "drug/TRIKAFTA"
    assert top.fields["latest_patent_expiry"] == max(e.fields["expires"] for e in patents)
    assert top.fields["latest_exclusivity_expiry"] == max(e.fields["expires"] for e in excl) == "2033-03-27"
    app = ids["N212273"]
    own = [e for e in patents if e.fields["appl"] == "N212273"]
    assert app.fields["latest_patent_expiry"] == max(e.fields["expires"] for e in own)
    assert f"Latest-expiring patent: {app.fields['latest_patent_no']}" in app.text
    # within an application, patent and exclusivity records come latest expiry first
    dates = [e.fields["expires"] for e in out.evidence if e.fields.get("appl") == "N212273" and "expires" in e.fields]
    assert dates == sorted(dates, reverse=True)


def test_cache_key_changes_with_tool_version():
    assert cache_key("orange_book", {"drug": "x"}, 1) != cache_key("orange_book", {"drug": "x"}, 2)
