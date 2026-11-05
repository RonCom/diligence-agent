"""Judge calibration: a blind labeling sheet of dev answers, and Cohen's kappa between the judge and the hand labels.

Pre-registered: 40 dev answers, all judge-graded ones plus a category-stratified sample; the judge counts only if
kappa >= 0.6 on correctness and on faithfulness, computed separately. The sheet shows the question, the reference,
the answer and the records it cites, and never the judge's or the code's grade, or which system answered: that
lives in a separate key file (label_key.json) that `calibrate` reads. A label counts only against the
exact answer it graded (answer_sha), so a rerun that changes an answer does not inherit its label.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from diligence_agent.evidence import citations

SYSTEMS = ("agent", "baseline", "langchain")
FIELDS = ["label_id", "question_id", "question", "reference", "answer", "cited_records",
          "human_correct", "human_faithful", "notes"]  # fmt: skip
RECORD_CHARS = 700


def answer_sha(answer: str) -> str:
    return hashlib.sha256(answer.encode()).hexdigest()[:12]


def sample(rows: list[dict], qs: dict[str, dict], n: int = 40, seed: int = 0) -> list[dict]:
    """Every judge-graded answer, then a round-robin over categories of the rest until n."""
    rng = random.Random(seed)
    judged = [r for r in rows if qs[r["id"]]["check"]["type"] == "judge"]
    rest = [r for r in rows if qs[r["id"]]["check"]["type"] != "judge"]
    rng.shuffle(rest)
    by_cat: dict[str, list[dict]] = {}
    for r in rest:
        by_cat.setdefault(r["category"], []).append(r)
    picked = list(judged)
    cats = sorted(by_cat)
    while len(picked) < n and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    rng.shuffle(picked)
    return picked


def sheet_rows(picked: list[dict], qs: dict[str, dict]) -> list[dict[str, str]]:
    out = []
    for i, r in enumerate(picked, 1):
        cited = [k for k in dict.fromkeys(citations(r["answer"]))]
        recs = "\n".join(
            f"[{k}] {r['records'].get(k, '(not among the records collected)')[:RECORD_CHARS]}" for k in cited
        )
        out.append({"label_id": f"L{i:02d}", "system": r["system"], "question_id": r["id"],
                    "answer_sha": answer_sha(r["answer"]), "question": r["question"],
                    "reference": qs[r["id"]]["reference"], "answer": r["answer"], "cited_records": recs or "(none)",
                    "human_correct": "", "human_faithful": "", "notes": ""})  # fmt: skip
    return out


GUIDE = """# Labeling guide: judge calibration

Label each answer in `eval/human_labels.csv` (opens in Excel). Fill two columns, and `notes` when useful.
Don't look up the judge's grades first: they are kept out of the sheet on purpose.

**human_correct**: 1 if the answer states the reference's key facts without contradicting them; 0.5 if it states
some but not all; 0 if it misses or contradicts them. For an out-of-scope question (reference NOT IN SOURCES), 1 only
if the answer declines. A decline on an answerable question is 0.

**human_faithful**: 1 if every claim in the answer is supported by the records it cites (shown below the answer);
0 if any claim is not. Faithful means supported by the records, not true: a wrong answer that quotes its records
accurately is faithful, and correctness is scored separately. A decline is faithful.

The judge counts only if Cohen's kappa against these labels is at least 0.6, on correctness and on faithfulness
separately. Below kappa 0.6, these labels replace the judge's correctness grades on the judge-graded questions.

After labeling, run `uv run diligence-agent calibrate`.

## The {n} answers
"""


LABEL_GUIDE = """# Labeling guide: judge-graded answers

The judge failed calibration (reports/judge_calibration.json), so by the pre-registered rule these hand labels
replace its correctness grades on the judge-graded questions. Label `human_correct` in the CSV beside this guide;
`human_faithful` is not used here. Same rubric as eval/LABELING.md: 1 if the answer states the reference's key
facts without contradicting them, 0.5 if some but not all, 0 if it misses or contradicts them; a decline on an
answerable question is 0. An identical answer to the same question appears once and its label covers every repeat.

After labeling, run `uv run diligence-agent regrade --split <split>` for each system.

## The {n} answers
"""


def key_path(csv_path: Path) -> Path:
    """label_key.json beside the calibration sheet; <stem>_key.json beside any other sheet."""
    return csv_path.with_name("label_key.json" if csv_path.stem == "human_labels" else f"{csv_path.stem}_key.json")


def judged_unlabeled(rows: list[dict], qs: dict[str, dict], labeled: set[tuple[str, str]]) -> list[dict]:
    """Answers to judge-checked questions with no hand label yet, one per distinct (question, answer text): repeats
    that produce the same answer share a label."""
    out, seen = [], set(labeled)
    for r in rows:
        k = (r["id"], answer_sha(r["answer"]))
        if qs[r["id"]]["check"]["type"] == "judge" and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def hand_labels(eval_dir: str | Path = "eval") -> dict[tuple[str, str], float]:
    """human_correct by (question id, answer sha) from every filled labeling sheet and its key file."""
    out: dict[tuple[str, str], float] = {}
    for p in sorted(Path(eval_dir).glob("human_labels*.csv")):
        if not key_path(p).exists():
            continue
        key = json.loads(key_path(p).read_text(encoding="utf8"))
        with open(p, encoding="utf-8-sig", newline="") as f:
            for lab in csv.DictReader(f):
                v = _num(lab.get("human_correct"))
                if v is not None and lab["label_id"] in key:
                    out[(lab["question_id"], key[lab["label_id"]]["answer_sha"])] = v
    return out


def labels_replace_judge(reports: str | Path = "reports") -> bool:
    """True once calibration has run and the judge fell below the pre-registered kappa."""
    p = Path(reports) / "judge_calibration.json"
    return p.exists() and not json.loads(p.read_text(encoding="utf8"))["judge_counts"]


def write_sheet(picked: list[dict], qs: dict[str, dict], csv_path: Path, guide_path: Path, guide: str = "") -> None:
    rows = sheet_rows(picked, qs)
    key = {r["label_id"]: {"system": r.pop("system"), "answer_sha": r.pop("answer_sha")} for r in rows}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            if any(r.get("human_correct") or r.get("human_faithful") for r in csv.DictReader(f)):
                raise RuntimeError(f"{csv_path} already has labels; refusing to overwrite")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    key_path(csv_path).write_text(json.dumps(key, indent=1), encoding="utf8", newline="\n")
    parts = [(guide or GUIDE).format(n=len(rows))]
    for r in rows:
        parts.append(f"### {r['label_id']} ({r['question_id']})\n\n**Question:** {r['question']}\n\n"
                     f"**Reference:** {r['reference']}\n\n**Answer:**\n\n> {r['answer']}\n\n"
                     f"**Cited records:**\n\n```\n{r['cited_records']}\n```\n")  # fmt: skip
    guide_path.write_text("\n".join(parts), encoding="utf8", newline="\n")


# ---- kappa ------------------------------------------------------------------------------------------------------


def cohen_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa for two raters over the same items; None when undefined (no items or one shared label)."""
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _num(x) -> float | None:
    try:
        return None if x in ("", None) else float(x)
    except (TypeError, ValueError):
        return None


def calibrate(labels_csv: Path, answers: dict[str, list[dict]], kappa_min: float = 0.6) -> dict[str, Any]:
    with open(labels_csv, encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    key = json.loads(key_path(labels_csv).read_text(encoding="utf8"))
    for lab in labels:
        lab |= key[lab["label_id"]]
    index = {(r["system"], r["id"]): r for rows in answers.values() for r in rows}
    matched, stale = [], []
    for lab in labels:
        r = index.get((lab["system"], lab["question_id"]))
        if r is None or answer_sha(r["answer"]) != lab["answer_sha"]:
            stale.append(lab["label_id"])
            continue
        matched.append((lab, r))

    def block(human_key: str, judge_key: str, passing) -> dict[str, Any]:
        pairs = [(passing(_num(lab[human_key])), passing(_num(r.get(judge_key)))) for lab, r in matched
                 if _num(lab[human_key]) is not None and _num(r.get(judge_key)) is not None]  # fmt: skip
        h, j = [p[0] for p in pairs], [p[1] for p in pairs]
        k = cohen_kappa(h, j)
        return {"n": len(pairs), "agreement": (sum(x == y for x, y in pairs) / len(pairs)) if pairs else None,
                "cohen_kappa": k, "human_pass_rate": (sum(h) / len(h)) if h else None,
                "judge_pass_rate": (sum(j) / len(j)) if j else None,
                "passes": k is not None and k >= kappa_min}  # fmt: skip

    correct = block("human_correct", "judge_correct", lambda v: v == 1.0)
    faithful = block("human_faithful", "judge_faithful", lambda v: v == 1.0)
    every = [r for rows in answers.values() for r in rows]
    code_rows = [
        r for r in every if r.get("check_type") not in ("judge", None) and _num(r.get("judge_correct")) is not None
    ]
    code = [(_num(r["correct_raw"]) == 1.0, _num(r["judge_correct"]) == 1.0) for r in code_rows]
    return {
        "labels": len(labels), "matched": len(matched), "stale_labels": stale, "kappa_min": kappa_min,
        "correctness": correct, "faithfulness": faithful,
        "judge_counts": correct["passes"] and faithful["passes"],
        "judge_vs_code": {"n": len(code), "agreement": (sum(x == y for x, y in code) / len(code)) if code else None,
                          "cohen_kappa": cohen_kappa([c[0] for c in code], [c[1] for c in code])},
    }  # fmt: skip


def load_answers(split: str = "dev", reports: str | Path = "reports") -> dict[str, list[dict]]:
    """Every saved answer per system; on test, every repeat's."""
    out = {}
    for s in SYSTEMS:
        pattern = f"answers_{s}_{split}.jsonl" if split == "dev" else f"answers_{s}_{split}_r*.jsonl"
        rows = [json.loads(line) for p in sorted(Path(reports).glob(pattern))
                for line in open(p, encoding="utf8") if line.strip()]  # fmt: skip
        if rows:
            out[s] = rows
    return out
