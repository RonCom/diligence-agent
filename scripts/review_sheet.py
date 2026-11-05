"""Write eval/REVIEW.md: a sample of questions with gold answers and links to the source records, for a hand check.

uv run python scripts/review_sheet.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

from diligence_agent import evalset

TENK_URL = "https://www.sec.gov/Archives/edgar/data/1802768/000180276826000003/rprx-20251231.htm"
SAMPLE_PER_CATEGORY = 2
SEED = 11


def url(key: str) -> str:
    source, _, rid = key.partition(":")
    if source == "orange_book":
        m = re.match(r"([NA])(\d+)", rid)
        return f"https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type={m[1]}&Appl_No={m[2]}"
    if source == "drugs_at_fda":
        digits = "".join(c for c in rid.split("/")[0] if c.isdigit())
        return f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={digits}"
    if source in ("fda_label", "label_text"):
        return f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={re.split(r'[/@]', rid)[0]}"
    if source == "ctgov":
        return f"https://clinicaltrials.gov/study/{rid}"
    if source == "sec_filing":
        return TENK_URL
    return ""


def gold_text(q: dict) -> str:
    return q["reference"].replace("|", "/")


def links(q: dict) -> str:
    urls = list(dict.fromkeys(u for u in (url(k) for k in q["sources"]) if u))
    if urls:
        return " ".join(f"[{i + 1}]({u})" for i, u in enumerate(urls[:3]))
    return "Orange Book has no listing" if q["category"] == "biologic" else "none (should decline)"


def main(path: str = "eval/questions.jsonl", out: str = "eval/REVIEW.md") -> None:
    qs = evalset.load(path)
    rng = random.Random(SEED)
    sample = []
    for cat in dict.fromkeys(q["category"] for q in qs):
        items = [q for q in qs if q["category"] == cat]
        sample += rng.sample(items, min(SAMPLE_PER_CATEGORY, len(items)))
    lines = [
        "# Eval set review",
        "",
        f"{len(qs)} questions, built {qs[0]['as_of']} by `diligence-agent build-eval`; gold answers come from the "
        "sources in code. Check the sample below against the linked records and mark each row. Fix wording or drop "
        "a question in `src/diligence_agent/evalset.py`, then rebuild.",
        "",
        f"## Sample to check ({len(sample)})",
        "",
        "| OK? | ID | Split | Question | Gold | Check | Sources |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in sample:
        lines.append(f"| [ ] | {q['id']} | {q['split']} | {q['question']} | {gold_text(q)} | {q['check']['type']} | "
                     f"{links(q)} |")  # fmt: skip
    lines += ["", "## All questions", ""]
    for cat in dict.fromkeys(q["category"] for q in qs):
        lines += [f"### {cat}", "", "| ID | Split | Style | Question | Gold | Check |", "|---|---|---|---|---|---|"]
        for q in (q for q in qs if q["category"] == cat):
            lines.append(f"| {q['id']} | {q['split']} | {q['style']} | {q['question']} | {gold_text(q)} | "
                         f"{q['check']['type']} |")  # fmt: skip
        lines.append("")
    Path(out).write_text("\n".join(lines), encoding="utf8", newline="\n")
    print(f"wrote {out}: {len(sample)} sampled of {len(qs)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
