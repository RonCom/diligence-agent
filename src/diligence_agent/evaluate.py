"""Run a system over an eval split, grade every answer in code, and summarize with bootstrap intervals.

Answers are generated first and the judge grades afterwards (`judge` command), so the 26b judge and the e4b answer
model do not swap in and out of an 8 GB GPU per question. Each run is an MLflow run with lineage parameters and one
trace per question; answers and grades land in reports/answers_<system>_<split>.jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from diligence_agent import calibrate, evalset, grading, prereg, tracing
from diligence_agent.scoring import mean_ci
from diligence_agent.tools.base import ToolRunner

SYSTEMS = ("agent", "baseline", "langchain")
AGENT_EXTRAS = (
    "claims",
    "claims_supported",
    "claims_cut",
    "loop_stops",
    "repair_attempts",
    "research_rounds",
    "planner_out_of_scope",
    "role_ms",
)
RECORD_CHARS = 2000


def cite_check(answer: str, records: dict[str, str], question: str = "") -> dict[str, Any]:
    """Faithfulness from the verifier's code check, run the same way on every system's saved answer: a claim passes
    when it cites only records the system collected and every date, year and percentage it states is in them or in
    the question. Decline sentences and lead-ins ending in a colon make no claim. An answer is faithful when every
    claim passes."""
    from diligence_agent.agent import code_check_claims

    claims = [c for c in code_check_claims(answer, records, given=question)
              if not grading.declined(c.text) and not c.text.rstrip().endswith(":")]  # fmt: skip
    ok = sum(c.code_ok for c in claims)
    return {"cite_claims": len(claims), "cite_claims_ok": ok,
            "cite_faithful": 1.0 if ok == len(claims) else 0.0,
            "cite_failures": [c.text[:200] for c in claims if not c.code_ok]}  # fmt: skip


def human_for(q: dict, answer: str, labels: dict | None) -> dict | None:
    """The hand label to grade with, or None while the judge's grades count (labels is None)."""
    if labels is None or q["check"]["type"] != "judge":
        return None
    return {"correct": labels.get((q["id"], calibrate.answer_sha(answer)))}


def load_labels(reports: str | Path = "reports", eval_dir: str | Path = "eval") -> dict | None:
    return calibrate.hand_labels(eval_dir) if calibrate.labels_replace_judge(reports) else None


def answers_path(system: str, split: str, reports: str | Path = "reports", repeat: int = 1) -> Path:
    """Dev has one answers file per system; each test repeat has its own (r1 is primary, r2 measures rerun change)."""
    name = f"answers_{system}_{split}.jsonl" if split == "dev" else f"answers_{system}_{split}_r{repeat}.jsonl"
    return Path(reports) / name


def _row(q: dict, system: str, answer: str, records: dict[str, str], s: dict, labels=None) -> dict[str, Any]:
    calls = s.get("tool_calls", [])
    records = {k: v[:RECORD_CHARS] for k, v in records.items()}
    return {
        "id": q["id"], "category": q["category"], "style": q["style"], "split": q["split"],
        "answerable": q["answerable"], "system": system, "question": q["question"], "answer": answer,
        "records": records,
        "tool_calls": len(calls),
        "tool_call_validity": (sum(c["ok"] for c in calls) / len(calls)) if calls else None,
        "model_calls": s.get("model_calls"), "repeated_calls": s.get("repeated_calls", 0),
        "budget_stop": s.get("budget_stop", False), "citations": s.get("citations", 0),
        "citations_to_unseen_records": s.get("citations_to_unseen_records", []),
        "input_tokens": s.get("input_tokens", 0), "output_tokens": s.get("output_tokens", 0),
        "cost_usd": s.get("cost_usd", 0.0), "latency_s": s.get("latency_s"), "trace_id": s.get("trace_id"),
    } | {k: s[k] for k in AGENT_EXTRAS if k in s} | grading.grade(q, answer, human=human_for(q, answer, labels)) \
        | cite_check(answer, records, q["question"])  # fmt: skip


def answer_one(system: str, q: dict, ctx: dict) -> dict:
    if system == "baseline":
        from diligence_agent import baseline

        r = baseline.run_question(q["question"], ctx["index"], ctx["model"], ctx["cfg"], tags=ctx["tags"])
        return _row(q, system, r.answer, r.records, r.summary(), ctx.get("labels"))
    if system == "agent":
        from diligence_agent import agent

        r = agent.run_question(q["question"], ctx["runner"], ctx["cfg"], chat=ctx.get("model"), tags=ctx["tags"])
        evidence = {k: e.text for k, e in r.evidence.items()}
        return _row(q, system, r.answer, evidence, r.summary(), ctx.get("labels"))
    if system == "langchain":
        from diligence_agent import lc_agent

        r = lc_agent.run_question(q["question"], ctx["runner"], ctx["cfg"], model=ctx.get("model"), tags=ctx["tags"])
        records = {k: e.text for k, e in r.evidence.items()}
        return _row(q, system, r.answer, records, r.summary(), ctx.get("labels"))
    raise ValueError(system)


def run(
    system: str,
    split: str,
    cfg: dict,
    runner: ToolRunner,
    fake: bool = False,
    limit: int | None = None,
    model=None,
    ids: list[str] | None = None,
    reports: str | Path = "reports",
    log=print,
    repeat: int = 1,
) -> list[dict]:
    ok, why = prereg.can_score(cfg, split)
    if not ok:
        raise RuntimeError(f"refusing to score {split}: {why}")
    out = answers_path(system, split, reports, repeat)
    if split == "test" and out.exists():
        raise RuntimeError(f"refusing to score test again: {out} exists (each repeat is run once)")
    qs = [q for q in evalset.load(cfg["eval"]["questions"]) if q["split"] == split]
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    qs = qs[:limit] if limit else qs
    ctx: dict[str, Any] = {"cfg": cfg, "runner": runner, "model": model, "labels": load_labels(reports)}
    prompts: dict[str, str] = {}
    if system == "baseline":
        from diligence_agent import baseline
        from diligence_agent.lc_agent import build_model

        ctx["index"] = baseline.load_index(cfg, fake=fake)
        ctx["model"] = model or build_model(cfg)
        prompts["baseline"] = baseline.SYSTEM_PROMPT
    elif system == "agent":
        from diligence_agent import agent

        prompts |= {
            "planner": agent.PLANNER,
            "researcher": agent.RESEARCHER,
            "writer": agent.WRITER,
            "verifier": agent.VERIFIER,
        }
    else:
        from diligence_agent.lc_agent import SYSTEM_PROMPT

        prompts["lc_agent"] = SYSTEM_PROMPT
    lineage = tracing.lineage(cfg, prompts, fake=fake) | {"system": system, "split": split, "repeat": repeat,
                                                          "questions": len(qs),
                                                          "eval_set_sha": tracing.sha(Path(cfg["eval"]["questions"])
                                                                                      .read_bytes())}  # fmt: skip
    ctx["tags"] = lineage
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=f"{system}-{split}" + (f"-r{repeat}" if split == "test" else "")):
        mlflow.log_params({k: str(v)[:500] for k, v in lineage.items()})
        for i, q in enumerate(qs, 1):
            t0 = time.perf_counter()
            row = answer_one(system, q, ctx) | {"repeat": repeat}
            rows.append(row)
            log(f"[{i}/{len(qs)}] {q['id']} {row['check_type']:12} strict={row['correct_strict']} "
                f"{time.perf_counter() - t0:5.1f}s  {row['answer'][:90]!r}")  # fmt: skip
        write_rows(rows, out)
        summary = summarize(rows, cfg)
        mlflow.log_metrics({k: v for k, v in flat_metrics(summary).items() if v is not None})
        mlflow.log_artifact(str(out))
    log(json.dumps(headline(summary), indent=1))
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_rows(
    rows: list[dict], qs: dict[str, dict], chat, faithfulness: bool = False, log=print, labels: dict | None = None
) -> list[dict]:
    """Judge the judge-checked questions (and every answer's faithfulness if asked); re-grade with the code rules."""
    out = []
    for r in rows:
        q = qs[r["id"]]
        if q["check"]["type"] == "judge" or faithfulness:
            records = "\n".join(f"[{k}] {v}" for k, v in r["records"].items())
            judged = grading.judge(chat, q, r["answer"], records)
            r = r | grading.grade(q, r["answer"], judged, human=human_for(q, r["answer"], labels))
            log(f"{r['id']}: judge correct={judged['judge_correct']} faithful={judged['judge_faithful']}")
        out.append(r)
    return out


def regrade(rows: list[dict], qs: dict[str, dict], labels: dict | None = None) -> list[dict]:
    """Re-run the code checks, the citation check and the rules on saved answers, keeping any judge grades given;
    with labels (the judge failed calibration), hand labels grade the judge-checked questions."""
    out = []
    for r in rows:
        q = qs[r["id"]]
        judged = {k: r.get(k) for k in ("judge_correct", "judge_faithful", "judge_reason")}
        judged["judge_faithful"] = r.get("judge_faithful_raw")
        g = grading.grade(q, r["answer"], judged if r.get("judge_correct") is not None else None,
                          human=human_for(q, r["answer"], labels))  # fmt: skip
        out.append(r | g | cite_check(r["answer"], r["records"], q["question"]))
    return out


def _pct(values, q):
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def summarize(rows: list[dict], cfg: dict) -> dict[str, Any]:
    b = cfg.get("eval", {}).get("bootstrap", {})
    kw = {"resamples": b.get("resamples", 10_000), "seed": b.get("seed", 0), "ci": b.get("ci", 0.95)}
    answerable = [r for r in rows if r["answerable"] and r["check_type"] != "none_listed"]
    oos = [r for r in rows if not r["answerable"]]
    out: dict[str, Any] = {
        "questions": len(rows),
        "pending_judge": sum(r["correct_strict"] is None for r in rows),
        "pending_hand_labels": sum(r.get("correct_source") == "human" and r["correct_raw"] is None for r in rows),
        "correct_source": {s: sum(r.get("correct_source") == s for r in rows) for s in ("code", "judge", "human")},
        "cite_faithful": mean_ci([r.get("cite_faithful") for r in rows if not r["declined"]], **kw),
        "cite_claims": sum(r.get("cite_claims", 0) for r in rows),
        "cite_claims_ok": sum(r.get("cite_claims_ok", 0) for r in rows),
        "correct_strict": mean_ci([r["correct_strict"] for r in rows], **kw),
        "correct_raw": mean_ci([r["correct_raw"] for r in rows], **kw),
        "by_category": {c: mean_ci([r["correct_strict"] for r in rows if r["category"] == c], **kw)
                        for c in dict.fromkeys(r["category"] for r in rows)},
        "by_style": {s: mean_ci([r["correct_strict"] for r in rows if r["style"] == s], **kw)
                     for s in ("direct", "plain")},
        "false_declines": sum(r["declined"] for r in answerable),
        "answerable": len(answerable),
        "out_of_scope_declined": sum(r["declined"] for r in oos),
        "out_of_scope": len(oos),
        "cited_unseen_records": sum(bool(r["citations_to_unseen_records"]) for r in rows),
        "answers_with_citations": sum(r["citations"] > 0 for r in rows if not r["declined"]),
        "answers_not_declined": sum(not r["declined"] for r in rows),
        "tool_call_validity": mean_ci([r["tool_call_validity"] for r in rows], **kw),
        "model_calls_mean": float(np.mean([r["model_calls"] or 0 for r in rows])) if rows else None,
        "budget_stops": sum(bool(r["budget_stop"]) for r in rows),
        "repeated_calls": sum(r["repeated_calls"] or 0 for r in rows),
        "latency_p50_s": _pct([r["latency_s"] for r in rows], 50),
        "latency_p95_s": _pct([r["latency_s"] for r in rows], 95),
        "tokens_mean": float(np.mean([r["input_tokens"] + r["output_tokens"] for r in rows])) if rows else None,
        "cost_usd_total": float(sum(r["cost_usd"] for r in rows)),
    }  # fmt: skip
    return out


def headline(s: dict) -> dict:
    c = s["correct_strict"]
    f = lambda x: None if x is None else round(x, 3)  # noqa: E731
    return {
        "correct_strict": f"{f(c['mean'])} [{f(c['lo'])}, {f(c['hi'])}] n={c['n']}",
        "pending_judge": s["pending_judge"],
        "pending_hand_labels": s["pending_hand_labels"],
        "correct_source": s["correct_source"],
        "cite_faithful": f"{f(s['cite_faithful']['mean'])} [{f(s['cite_faithful']['lo'])}, "
        f"{f(s['cite_faithful']['hi'])}] n={s['cite_faithful']['n']}",
        "cite_claims_ok": f"{s['cite_claims_ok']} / {s['cite_claims']}",
        "by_category": {k: f(v["mean"]) for k, v in s["by_category"].items()},
        "by_style": {k: f(v["mean"]) for k, v in s["by_style"].items()},
        "false_declines": f"{s['false_declines']} / {s['answerable']}",
        "out_of_scope_declined": f"{s['out_of_scope_declined']} / {s['out_of_scope']}",
        "cited_unseen_records": s["cited_unseen_records"],
        "latency_p50_p95_s": [f(s["latency_p50_s"]), f(s["latency_p95_s"])],
        "tokens_mean": f(s["tokens_mean"]),
        "cost_usd_total": s["cost_usd_total"],
    }


def flat_metrics(s: dict) -> dict[str, float | None]:
    out = {"correct_strict": s["correct_strict"]["mean"], "correct_strict_lo": s["correct_strict"]["lo"],
           "correct_strict_hi": s["correct_strict"]["hi"], "correct_raw": s["correct_raw"]["mean"],
           "false_declines": s["false_declines"], "out_of_scope_declined": s["out_of_scope_declined"],
           "latency_p50_s": s["latency_p50_s"], "latency_p95_s": s["latency_p95_s"],
           "tokens_mean": s["tokens_mean"], "cost_usd_total": s["cost_usd_total"],
           "pending_judge": s["pending_judge"], "pending_hand_labels": s["pending_hand_labels"],
           "cite_faithful": s["cite_faithful"]["mean"]}  # fmt: skip
    out |= {f"correct_{k}": v["mean"] for k, v in s["by_category"].items()}
    return out
