"""Build the eval set: questions with gold answers computed from the systems of record.

No gold value is typed by hand. Each one comes from a tool call or the downloaded 10-K at build time, and the
builder fails if a source no longer supports a question (a keyword missing from a label, a 10-K phrase not found).
`diligence-agent build-eval --check` recomputes every gold answer and lists the ones that changed, which matters for
ClinicalTrials.gov: statuses and completion dates move, so gold is re-verified before the frozen test run.

Categories and checks:
  patent       latest Orange Book patent expiry           date, code
  exclusivity  latest Orange Book exclusivity expiry      date, code
  approval     first FDA approval (Drugs@FDA ORIG)        date, code
  indication   is X approved for Y (label text)           yes/no, code
  trial        status and primary completion of an NCT    status + month, code
  readouts     Phase 3 trials completing within 12 months NCT set F1, code
  tenk         10-K royalty terms, receipts, sales        numbers and years in code; rate wording and footnotes by judge
  conflict     10-K estimate vs Orange Book date          both years, code
  biologic     Orange Book patents for a BLA product      "none listed" or a decline, code
  out_of_scope nothing in the sources answers it          decline, code
"""

from __future__ import annotations

import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from diligence_agent.corpus import html_lines, sectioned
from diligence_agent.tools.base import ToolRunner

AS_OF = date(2026, 11, 4)
ACTIVE = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
TENK_HTML = "data/raw/edgar/rprx-20251231.htm"
DEV_SHARE = 1 / 3
SEED = 7

# ---- question specs: what to ask; the gold comes from the sources -------------------------------------------------

PATENT = ["Xtandi", "Trikafta", "Alyftrek", "Kalydeco", "Imbruvica", "Evrysdi", "Erleada", "Voranigo", "Cabometyx",
          "Spinraza", "Amvuttra", "Farxiga"]  # fmt: skip
EXCLUSIVITY = ["Trikafta", "Alyftrek", "Voranigo", "Myqorzo", "Orladeyo", "Evrysdi", "Amvuttra", "Imbruvica"]
APPROVAL = ["Tysabri", "Xtandi", "Tremfya", "Trodelvy", "Imdelltra", "Voranigo", "Myqorzo", "Kalydeco", "Promacta",
            "Spinraza"]  # fmt: skip
INDICATION = [  # (drug, condition as asked, label keyword, expected)
    ("Tremfya", "Crohn's disease", "crohn", True),
    ("Xtandi", "non-metastatic castration-sensitive prostate cancer with biochemical recurrence",
     "biochemical recurrence", True),
    ("Trelegy Ellipta", "asthma", "asthma", True),
    ("Trodelvy", "triple-negative breast cancer", "triple negative", True),
    ("Amvuttra", "the cardiomyopathy of transthyretin-mediated amyloidosis", "cardiomyopathy", True),
    ("Tysabri", "ulcerative colitis", "ulcerative colitis", False),
    ("Spinraza", "amyotrophic lateral sclerosis (ALS)", "amyotrophic", False),
    ("Erleada", "non-metastatic castration-sensitive prostate cancer", "non-metastatic castration-sensitive", False),
    ("Imdelltra", "limited-stage small cell lung cancer", "limited stage", False),
    ("Voranigo", "glioblastoma", "glioblastoma", False),
]  # fmt: skip
TRIALS = [("NCT06211036", "Imdelltra"), ("NCT05067790", "Evrysdi"), ("NCT05347095", "Tremfya"),
          ("NCT04455750", "Xtandi"), ("NCT02531516", "Erleada"), ("NCT05254743", "Imbruvica")]  # fmt: skip
READOUTS = [("Evrysdi", "risdiplam"), ("Imdelltra", "tarlatamab"), ("Spinraza", "nusinersen"),
            ("Erleada", "apalutamide"), ("Tremfya", "guselkumab"), ("Trodelvy", "sacituzumab govitecan")]  # fmt: skip
TENK_RATE = ["Xtandi", "Tremfya", "Voranigo", "Trodelvy", "Tysabri", "Imdelltra"]
TENK_DURATION = ["Evrysdi", "Promacta", "Tremfya", "Erleada"]
TENK_RECEIPTS = ["Tysabri", "Trelegy", "Spinraza", "Imbruvica"]
TENK_SALES = ["Xtandi", "Tremfya", "Imbruvica"]
TENK_YEAR = ["Tysabri", "Trodelvy"]
TENK_FOOTNOTES = [  # (id suffix, question, plain question, phrases that must be in the 10-K, key facts, check tokens)
    ("alyftrek", "What royalty rate does Royalty Pharma expect on Alyftrek, and what is in dispute about it?",
     "Is Royalty Pharma's Alyftrek royalty contested, and what's at stake?",
     ["deuterated ivacaftor (deutivacaftor) is the same as ivacaftor", "approximately 8% for Alyftrek",
      "approximately 4% for Alyftrek"],
     ["Royalty Pharma believes deutivacaftor is royalty-bearing, giving a blended royalty of about 8%",
      "Vertex says deutivacaftor is not royalty-bearing, which would give about 4%"], None),
    ("spinraza", "When does Royalty Pharma's royalty interest in Spinraza revert to Ionis?", None,
     ["revert to Ionis after we receive aggregate Spinraza royalties equal to $475 million or $550 million"],
     ["It reverts to Ionis after Royalty Pharma receives aggregate Spinraza royalties of $475 million or "
      "$550 million, depending on the timing and occurrence of certain events"], None),
    ("cabometyx", "Until when does Royalty Pharma receive royalties on U.S. sales of cabozantinib products?", None,
     ["royalties on U.S. sales of cabozantinib products through September 2026"],
     ["Through September 2026 for U.S. sales; non-U.S. markets through the full term of the royalty"],
     ["September", "2026"]),
    ("alyftrek-ec", "When did the European Commission approve Alyftrek, according to Royalty Pharma's 10-K?",
     "When did Alyftrek get its EU approval?",
     ["In July 2025, Vertex announced that the European Commission"],
     ["July 2025"], ["July", "2025"]),
]  # fmt: skip
CONFLICT = [  # (drug, question, 10-K phrase, 10-K year)
    ("Trikafta", "When do Trikafta's patents expire? Compare Royalty Pharma's 10-K with the FDA Orange Book.",
     "We estimate expected Trikafta patent expiration in 2037", "2037"),
    ("Alyftrek", "Royalty Pharma ties its cystic fibrosis royalty duration to Alyftrek's patents. What does its 10-K "
     "estimate, and when does Alyftrek's last Orange Book patent expire?",
     "We estimate royalty duration of 2039-2041 due to expected Alyftrek patent expiration", "2039"),
]  # fmt: skip
BIOLOGIC = ["Tremfya", "Tysabri", "Trodelvy"]
OUT_OF_SCOPE = [  # (question, style)
    ("What is the U.S. list price (WAC) for a month of Trikafta?", "direct"),
    ("What's the analyst consensus price target for Royalty Pharma stock?", "plain"),
    ("What royalty rate does Royalty Pharma receive on Ozempic?", "direct"),
    ("What gross margin does Vertex earn on Trikafta?", "direct"),
    ("When does the EU supplementary protection certificate for Trikafta expire?", "direct"),
    ("What are analysts forecasting for Voranigo's peak sales?", "plain"),
    ("What will Royalty Pharma's total revenue be in 2027?", "direct"),
    ("How much did Merck report in worldwide Keytruda sales for 2025?", "direct"),
    ("How much did Medicare Part D spend on Tremfya in 2024?", "direct"),
    ("When did the European Commission approve Voranigo?", "direct"),
    ("What's Imbruvica's U.S. market share in CLL right now?", "plain"),
    ("Who is the chief executive of Servier?", "direct"),
    ("How many units of Xtandi did Astellas ship in the third quarter of 2026?", "direct"),
    ("What did Royalty Pharma's board decide at its December 2026 meeting?", "direct"),
    ("What does a course of Evrysdi cost in Japan?", "plain"),
]  # fmt: skip

# plain-language wording, the way an analyst would ask; used for every third question in a category
PLAIN = {
    "patent": "When does {drug} lose its last Orange Book patent protection?",
    "exclusivity": "How long does {drug}'s FDA exclusivity run?",
    "approval": "When did {drug} first get the FDA nod?",
    "indication": "Can {drug} be prescribed on-label for {condition}?",
    "trial": "Where does the {drug} study {nct} stand, and when should it read out?",
    "readouts": "Any {drug} Phase 3 readouts coming in the next 12 months?",
    "tenk_rate": "What cut of {drug} sales does Royalty Pharma get?",
    "tenk_duration": "How long will Royalty Pharma keep collecting on {drug}?",
    "tenk_receipts": "How much royalty cash did {drug} bring Royalty Pharma in 2025?",
    "tenk_sales": "How big was {drug} in 2025 sales terms?",
    "tenk_year": "When did Royalty Pharma buy into {drug}?",
    "biologic": "When do {drug}'s Orange Book patents run out?",
}
DIRECT = {
    "patent": "When does the last-expiring FDA Orange Book patent listed for {drug} expire?",
    "exclusivity": "What is the latest-expiring FDA regulatory exclusivity listed for {drug} in the Orange Book, "
    "and when does it expire?",
    "approval": "On what date did the FDA first approve {drug}?",
    "indication": "Is {drug} FDA-approved for {condition}?",
    "trial": "What is the current overall status and primary completion date of {nct}?",
    "readouts": "Which Phase 3 trials of {generic} that are recruiting, active or not yet recruiting have a primary "
    "completion date between {start} and {end}? List their NCT numbers.",
    "tenk_rate": "What royalty rate does Royalty Pharma receive on {drug}, according to its latest 10-K?",
    "tenk_duration": "What is Royalty Pharma's estimated royalty duration for {drug}?",
    "tenk_receipts": "What were Royalty Pharma's 2025 Portfolio Receipts from {drug}?",
    "tenk_sales": "What were {drug}'s 2025 end-market sales, as reported in Royalty Pharma's 10-K?",
    "tenk_year": "In what year did Royalty Pharma acquire its royalty on {drug}?",
    "biologic": "When does the last-expiring FDA Orange Book patent listed for {drug} expire?",
}


def wording(kind: str, i: int, **kw) -> tuple[str, str]:
    style = "plain" if i % 3 == 1 and kind in PLAIN else "direct"
    return (PLAIN if style == "plain" else DIRECT)[kind].format(**kw), style


# ---- gold helpers --------------------------------------------------------------------------------------------------


def _call(runner: ToolRunner, tool: str, args: dict):
    out = runner.call(tool, args)
    if not out.ok:
        raise RuntimeError(f"{tool}({args}) failed: {out.error}")
    return out.evidence


def _latest(evidence, field: str, kind: str):
    """(value, expires, record key) for the latest patent ('patent_no') or exclusivity ('code')."""
    recs = [e for e in evidence if kind in e.fields and e.fields.get("expires") and not e.fields.get("delisted")]
    if not recs:
        return None
    best = max(recs, key=lambda e: e.fields["expires"])
    return best.fields[kind], best.fields["expires"], best.key


def tenk_tables(html_path: str | Path) -> tuple[dict[str, list[str]], dict[str, list[str]], str]:
    """Rows of the Portfolio Summary and Portfolio Overview tables, keyed by product, plus the full 10-K text."""
    secs = dict(sectioned(html_lines(Path(html_path).read_bytes())))

    def rows(section: str) -> dict[str, list[str]]:
        out = {}
        for line in secs[section]:
            if " | " not in line:
                continue
            cells = [c.strip() for c in line.split(" | ")]
            name = re.sub(r"\s*\(\d+\)$", "", cells[0]).strip()
            out[name] = cells
        return out

    text = "\n".join(line for _, body in secs.items() for line in body)
    return rows("Item 1. BUSINESS > Portfolio Summary"), rows("Item 1. BUSINESS > Portfolio Overview"), text


def _money_millions(cell: str) -> float:
    return float(cell.replace("$", "").replace(",", "").strip())


def _chunk_with(index_dir: Path, needle: str) -> str | None:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return None
    for line in open(path, encoding="utf8"):
        c = json.loads(line)
        if needle in c["text"]:
            return f"{c['source']}:{c['chunk_id']}"
    return None


# ---- build -------------------------------------------------------------------------------------------------------


def build(runner: ToolRunner, data_dir: str | Path = "data", as_of: date = AS_OF) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    qs: list[dict[str, Any]] = []

    def add(qid, category, question, style, drug, answerable, check, gold, reference, sources, key_facts=None):
        qs.append(
            {
                "id": qid,
                "category": category,
                "style": style,
                "drug": drug,
                "question": question,
                "answerable": answerable,
                "check": check,
                "gold": gold,
                "reference": reference,
                "key_facts": key_facts or [reference],
                "sources": sources,
                "as_of": as_of.isoformat(),
            }
        )

    for i, drug in enumerate(PATENT):
        ev = _call(runner, "orange_book", {"drug": drug})
        pno, expires, key = _latest(ev, "expires", "patent_no")
        q, style = wording("patent", i, drug=drug)
        add(
            f"pat{i + 1:02d}",
            "patent",
            q,
            style,
            drug,
            True,
            {"type": "date", "value": expires},
            {"patent_no": pno, "expires": expires},
            f"Patent {pno} expires {expires}.",
            [key],
        )

    n = 0
    for drug in EXCLUSIVITY:
        latest = _latest(_call(runner, "orange_book", {"drug": drug}), "expires", "code")
        if latest is None or n == 6:
            continue
        code, expires, key = latest
        q, style = wording("exclusivity", n, drug=drug)
        n += 1
        add(
            f"exc{n:02d}",
            "exclusivity",
            q,
            style,
            drug,
            True,
            {"type": "date", "value": expires},
            {"code": code, "expires": expires},
            f"Exclusivity {code} expires {expires}.",
            [key],
        )

    for i, drug in enumerate(APPROVAL):
        ev = [e for e in _call(runner, "drugs_at_fda", {"drug": drug}) if e.fields["submission"].startswith("ORIG")]
        first = min(ev, key=lambda e: e.fields["approved"])
        q, style = wording("approval", i, drug=drug)
        add(
            f"app{i + 1:02d}",
            "approval",
            q,
            style,
            drug,
            True,
            {"type": "date", "value": first.fields["approved"]},
            {"application": first.fields["application"], "approved": first.fields["approved"]},
            f"{first.fields['application']} was approved {first.fields['approved']}.",
            [first.key],
        )

    for i, (drug, condition, keyword, expected) in enumerate(INDICATION):
        ev = _call(runner, "fda_label", {"drug": drug})
        text = " ".join(e.text for e in ev).lower().replace("-", " ").replace("‑", " ")
        found = keyword.replace("-", " ") in text
        if found != expected:
            raise RuntimeError(f"{drug} label: {keyword!r} found={found}, expected {expected}")
        q, style = wording("indication", i, drug=drug, condition=condition)
        ref = f"{'Yes' if expected else 'No'}: the current {drug} label {'lists' if expected else 'does not list'} it."
        add(
            f"ind{i + 1:02d}",
            "indication",
            q,
            style,
            drug,
            True,
            {"type": "yes_no", "value": expected},
            {"approved": expected, "keyword": keyword},
            ref,
            [e.key for e in ev[:1]],
        )

    for i, (nct, drug) in enumerate(TRIALS):
        (e,) = _call(runner, "ctgov_search", {"nct_ids": [nct]})
        q, style = wording("trial", i, nct=nct, drug=drug)
        gold = {"status": e.fields["status"], "primary_completion": e.fields["primary_completion"]}
        add(
            f"trl{i + 1:02d}",
            "trial",
            q,
            style,
            drug,
            True,
            {"type": "trial_status", **gold},
            gold,
            f"{nct} is {gold['status']}, primary completion {gold['primary_completion']}.",
            [e.key],
        )

    start, end = as_of, as_of + timedelta(days=365)
    for i, (drug, generic) in enumerate(READOUTS):
        ev = _call(
            runner,
            "ctgov_search",
            {
                "intervention": generic,
                "phase": "PHASE3",
                "status": ACTIVE,
                "primary_completion_from": start.isoformat(),
                "primary_completion_to": end.isoformat(),
                "max_results": 50,
            },
        )
        ids = sorted(e.record_id for e in ev)
        q, style = wording("readouts", i, drug=drug, generic=generic, start=start.isoformat(), end=end.isoformat())
        add(
            f"win{i + 1:02d}",
            "readouts",
            q,
            style,
            drug,
            True,
            {"type": "nct_set", "value": ids},
            {"nct_ids": ids, "window": [start.isoformat(), end.isoformat()], "query": generic},
            f"{len(ids)} trial{'s' if len(ids) != 1 else ''}: {', '.join(ids)}.",
            [e.key for e in ev],
        )

    summary, overview, tenk = tenk_tables(data_dir / "raw" / "edgar" / "rprx-20251231.htm")
    idx = data_dir / "index"
    k = 0

    def tenk_q(kind, drug, check, gold, reference, row_text, judge=False):
        nonlocal k
        k += 1
        q, style = wording(kind, k, drug=drug)
        add(
            f"tk{k:02d}",
            "tenk",
            q,
            style,
            drug,
            True,
            {"type": "judge"} if judge else check,
            gold,
            reference,
            [s for s in [_chunk_with(idx, row_text)] if s],
        )

    for drug in TENK_RATE:
        row = summary[drug]
        tenk_q(
            "tenk_rate",
            drug,
            None,
            {"rate": row[3]},
            f"{row[3]} (10-K Portfolio Summary).",
            " | ".join(row),
            judge=True,
        )
    for drug in TENK_DURATION:
        row = summary[drug]
        years = re.findall(r"\d{4}", row[2])
        tenk_q(
            "tenk_duration",
            drug,
            {"type": "contains_all", "value": years},
            {"duration": row[2]},
            f"Estimated royalty duration {row[2]}.",
            " | ".join(row),
        )
    for drug in TENK_RECEIPTS:
        row = overview[drug]
        v = _money_millions(row[-2])
        tenk_q(
            "tenk_receipts",
            drug,
            {"type": "money", "value": v},
            {"receipts_musd": v},
            f"${v:,.0f} million in 2025 Portfolio Receipts.",
            " | ".join(row),
        )
    for drug in TENK_SALES:
        row = overview[drug]
        v = _money_millions(row[-1])
        tenk_q(
            "tenk_sales",
            drug,
            {"type": "money", "value": v},
            {"sales_musd": v},
            f"${v:,.0f} million in 2025 end-market sales.",
            " | ".join(row),
        )
    for drug in TENK_YEAR:
        row = summary[drug]
        tenk_q(
            "tenk_year",
            drug,
            {"type": "contains_all", "value": [row[1]]},
            {"year": row[1]},
            f"Acquired in {row[1]}.",
            " | ".join(row),
        )
    for suffix, question, plain, phrases, facts, tokens in TENK_FOOTNOTES:
        flat = " ".join(tenk.split())
        missing = [p for p in phrases if p not in flat]
        if missing:
            raise RuntimeError(f"10-K no longer contains {missing}")
        k += 1
        style = "plain" if plain and k % 3 == 1 else "direct"
        add(
            f"tk{k:02d}",
            "tenk",
            plain if style == "plain" else question,
            style,
            suffix.split("-")[0].capitalize(),
            True,
            {"type": "contains_all", "value": tokens} if tokens else {"type": "judge"},
            {"facts": facts},
            " ".join(facts),
            [s for s in [_chunk_with(idx, phrases[0])] if s],
            key_facts=facts,
        )

    for i, (drug, question, phrase, year) in enumerate(CONFLICT):
        if phrase not in " ".join(tenk.split()):
            raise RuntimeError(f"10-K no longer says {phrase!r}")
        _, expires, key = _latest(_call(runner, "orange_book", {"drug": drug}), "expires", "patent_no")
        facts = [f"10-K: {phrase}", f"Orange Book: the last-expiring {drug} patent expires {expires}"]
        add(
            f"cnf{i + 1:02d}",
            "conflict",
            question,
            "direct",
            drug,
            True,
            {"type": "contains_all", "value": [year, expires[:4]]},
            {"tenk_year": year, "orange_book": expires},
            " ".join(facts),
            [key] + [s for s in [_chunk_with(idx, phrase)] if s],
            key_facts=facts,
        )

    for i, drug in enumerate(BIOLOGIC):
        if _call(runner, "orange_book", {"drug": drug}):
            raise RuntimeError(f"{drug} now has Orange Book records")
        q, style = wording("biologic", i, drug=drug)
        add(
            f"bla{i + 1:02d}",
            "biologic",
            q,
            style,
            drug,
            True,
            {"type": "none_listed"},
            {"listed": False},
            f"{drug} is a biologic (BLA); the Orange Book lists no patents for it.",
            [],
        )

    for i, (question, style) in enumerate(OUT_OF_SCOPE):
        add(
            f"oos{i + 1:02d}",
            "out_of_scope",
            question,
            style,
            None,
            False,
            {"type": "decline"},
            {},
            "NOT IN SOURCES",
            [],
        )

    assign_splits(qs)
    return qs


def assign_splits(qs: list[dict], dev_share: float = DEV_SHARE, seed: int = SEED) -> None:
    """Stratified by category: about a third of each category goes to dev, the rest to test."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for q in qs:
        by_cat.setdefault(q["category"], []).append(q)
    for items in by_cat.values():
        order = sorted(items, key=lambda q: q["id"])
        rng.shuffle(order)
        n_dev = max(1, round(len(order) * dev_share))
        for j, q in enumerate(order):
            q["split"] = "dev" if j < n_dev else "test"


def write(qs: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf8", newline="\n") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def diff(old: list[dict], new: list[dict]) -> list[str]:
    """Questions whose gold or check changed between two builds."""
    before = {q["id"]: q for q in old}
    out = []
    for q in new:
        prev = before.get(q["id"])
        if prev is None:
            out.append(f"{q['id']}: new")
        elif (prev["gold"], prev["check"]) != (q["gold"], q["check"]):
            out.append(f"{q['id']}: gold {prev['gold']} -> {q['gold']}")
    out += [f"{qid}: removed" for qid in before.keys() - {q["id"] for q in new}]
    return out
