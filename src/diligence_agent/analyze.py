"""The pre-registered tests (config.yaml `preregistered.tests`) on saved, graded answers, and run-to-run change.

Repeat 1 is primary: H1-H4 and every headline number come from it. Repeat 2, where present, is reported beside it:
how many answers changed between reruns and how far correct_strict moved. Nothing here re-grades; it reads
reports/answers_*.jsonl as `eval`, `judge` and `regrade` left them, and refuses while any grade is pending.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from diligence_agent import evaluate
from diligence_agent.scoring import mean_ci, paired_diff_ci

SYSTEMS = evaluate.SYSTEMS


def _kw(cfg: dict) -> dict:
    b = cfg.get("eval", {}).get("bootstrap", {})
    return {"resamples": b.get("resamples", 10_000), "seed": b.get("seed", 0), "ci": b.get("ci", 0.95)}


def load(split: str, reports: str | Path = "reports") -> dict[int, dict[str, list[dict]]]:
    """repeat -> system -> rows, for the files that exist."""
    out: dict[int, dict[str, list[dict]]] = {}
    for repeat in (1, 2) if split == "test" else (1,):
        for s in SYSTEMS:
            p = evaluate.answers_path(s, split, reports, repeat)
            if p.exists():
                out.setdefault(repeat, {})[s] = evaluate.load_rows(p)
    return out


def _false_declines(rows: list[dict]) -> int:
    return sum(r["declined"] for r in rows if r["answerable"] and r["check_type"] != "none_listed")


def _ratio(pass_if: str) -> tuple[int, int]:
    m = re.search(r"(\d+)\s+of\s+(\d+)", pass_if)
    if not m:
        raise ValueError(f"can't read a threshold from {pass_if!r}")
    return int(m[1]), int(m[2])


def run_tests(cfg: dict, rows: dict[str, list[dict]]) -> list[dict[str, Any]]:
    kw = _kw(cfg)
    score = {s: {r["id"]: r["correct_strict"] for r in rs} for s, rs in rows.items()}
    out = []
    for t in cfg["preregistered"]["tests"]:
        res: dict[str, Any] = {"id": t["id"], "claim": t["claim"], "pass_if": t["pass_if"]}
        if "compare" in t:
            a, b = t["compare"]
            if a not in rows or b not in rows:
                out.append(res | {"result": "missing system"})
                continue
            d = paired_diff_ci(score[a], score[b], **kw)
            res |= {"diff": d, "mean": {a: mean_ci(score[a].values(), **kw)["mean"],
                                        b: mean_ci(score[b].values(), **kw)["mean"]}}  # fmt: skip
            if t["id"] == "H3":
                if d["excludes_zero"]:
                    res["result"] = f"pass: {a if d['mean'] > 0 else b} is higher"
                else:
                    res["result"] = "no difference shown"
            else:
                fd = {a: _false_declines(rows[a]), b: _false_declines(rows[b])}
                res["false_declines"] = fd
                res["passed"] = bool(d["above_zero"] and fd[a] <= fd[b])
                res["result"] = "pass" if res["passed"] else "fail"
        else:  # H4: out-of-scope declines, per system
            need, of = _ratio(t["pass_if"])
            res["per_system"] = {}
            for s in t.get("applies_to", []):
                if s not in rows:
                    continue
                oos = [r for r in rows[s] if not r["answerable"]]
                got = sum(r["declined"] for r in oos)
                ok = got * of >= need * len(oos)  # "9 of 10" scaled to however many there are
                res["per_system"][s] = {"declined": got, "out_of_scope": len(oos), "passed": ok}
            res["passed"] = bool(res["per_system"]) and all(v["passed"] for v in res["per_system"].values())
            res["result"] = "pass" if res["passed"] else "fail"
        out.append(res)
    return out


def rerun_change(cfg: dict, r1: list[dict], r2: list[dict]) -> dict[str, Any]:
    a, b = {r["id"]: r for r in r1}, {r["id"]: r for r in r2}
    ids = sorted(a.keys() & b.keys())
    kw = _kw(cfg)
    return {
        "questions": len(ids),
        "answers_changed": sum(a[i]["answer"] != b[i]["answer"] for i in ids),
        "grade_changed": sum(a[i]["correct_strict"] != b[i]["correct_strict"] for i in ids),
        "correct_strict_r1": mean_ci([a[i]["correct_strict"] for i in ids], **kw),
        "correct_strict_r2": mean_ci([b[i]["correct_strict"] for i in ids], **kw),
        "r2_minus_r1": paired_diff_ci(
            {i: b[i]["correct_strict"] for i in ids}, {i: a[i]["correct_strict"] for i in ids}, **kw
        ),  # fmt: skip
    }


def analyze(cfg: dict, split: str = "test", reports: str | Path = "reports") -> dict[str, Any]:
    runs = load(split, reports)
    if 1 not in runs:
        raise RuntimeError(f"no {split} answers to analyze")
    pending = {f"{s} r{k}": n for k, rs in runs.items() for s, rows in rs.items()
               if (n := sum(r["correct_strict"] is None for r in rows))}  # fmt: skip
    if pending:
        raise RuntimeError(f"grades pending (judge or hand labels): {pending}")
    primary = runs[1]
    return {
        "split": split,
        "primary_repeat": 1,
        "systems": {s: evaluate.summarize(rows, cfg) for s, rows in primary.items()},
        "tests": run_tests(cfg, primary),
        "rerun": {s: rerun_change(cfg, primary[s], runs[2][s]) for s in primary if s in runs.get(2, {})},
        "preregistered": {
            "version": cfg["preregistered"].get("version"),
            "status": cfg["preregistered"]["status"],
            "test_tag": cfg["preregistered"].get("test_tag"),
        },  # fmt: skip
    }


def brief(res: dict) -> dict:
    f = lambda x: None if x is None else round(x, 3)  # noqa: E731
    out: dict[str, Any] = {}
    for s, summ in res["systems"].items():
        c = summ["correct_strict"]
        out[s] = f"{f(c['mean'])} [{f(c['lo'])}, {f(c['hi'])}] n={c['n']}"
    for t in res["tests"]:
        d = t.get("diff")
        ci = f" diff {f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]" if d else ""
        out[t["id"]] = t.get("result", "") + ci
    for s, r in res["rerun"].items():
        out[f"rerun {s}"] = f"{r['answers_changed']}/{r['questions']} answers changed, " \
                            f"{r['grade_changed']} grades changed"  # fmt: skip
    return out
