"""Command line: `tools`, `tool <name> <json args>` and `lc <question>`; add --fake to run offline."""

from __future__ import annotations

import argparse
import json
import sys

from diligence_agent.config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diligence-agent")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--fake", action="store_true", help="serve recorded tool responses; no network")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools", help="print the tool schemas the model sees")
    sub.add_parser("index", help="download filings and labels and build the doc_search index")
    t = sub.add_parser("tool", help="call one tool")
    t.add_argument("name")
    t.add_argument("args", nargs="?", default="{}", help="JSON arguments")
    ask = sub.add_parser("ask", help="answer a question with the custom multi-agent loop")
    ask.add_argument("question")
    lc = sub.add_parser("lc", help="answer a question with the LangChain tool-calling agent")
    lc.add_argument("question")
    be = sub.add_parser("build-eval", help="compute gold answers from the sources and write the eval set")
    be.add_argument("--out", default="eval/questions.jsonl")
    be.add_argument("--check", action="store_true", help="recompute gold and list changes; write nothing")
    sub.add_parser("index-baseline", help="build the baseline's strong index (docs plus structured records)")
    ev = sub.add_parser("eval", help="answer and code-grade one split with one system")
    ev.add_argument("--system", choices=["agent", "baseline", "langchain"], required=True)
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--ids", nargs="*", help="only these question ids")
    ev.add_argument("--repeat", type=int, default=1, help="test repeat number (1 is primary)")
    jd = sub.add_parser("judge", help="judge saved answers (judge-checked questions; --faithfulness for all)")
    jd.add_argument("--system", choices=["agent", "baseline", "langchain"], required=True)
    jd.add_argument("--split", choices=["dev", "test"], default="dev")
    jd.add_argument("--faithfulness", action="store_true")
    jd.add_argument("--repeat", type=int, default=1)
    ls = sub.add_parser("label-sheet", help="dev: the blind calibration sheet; test: judge-graded answers to label")
    ls.add_argument("--split", choices=["dev", "test"], default="dev")
    sub.add_parser("calibrate", help="kappa between the judge and the hand labels")
    rg = sub.add_parser("regrade", help="re-run the code checks on saved answers (after a grader fix)")
    rg.add_argument("--system", choices=["agent", "baseline", "langchain"], required=True)
    rg.add_argument("--split", choices=["dev", "test"], default="dev")
    rg.add_argument("--repeat", type=int, default=1)
    an = sub.add_parser("analyze", help="the pre-registered tests H1-H4 and run-to-run change -> reports/eval.json")
    an.add_argument("--split", choices=["dev", "test"], default="test")
    sub.add_parser("figures", help="README figures from reports/eval.json -> reports/figures/")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # labels contain characters such as "≥" that cp1252 can't print
        stream.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    from diligence_agent.tools import build_runner

    runner = build_runner(cfg, fake=args.fake, use_cache=args.cmd != "build-eval")  # gold from fresh data
    if args.cmd == "build-eval":
        from collections import Counter
        from pathlib import Path

        from diligence_agent import evalset

        qs = evalset.build(runner, cfg.get("data_dir", "data"))
        out = Path(args.out)
        if args.check:
            changes = evalset.diff(evalset.load(out), qs) if out.exists() else ["no existing eval set"]
            print("\n".join(changes) or "gold unchanged")
            return 1 if changes else 0
        evalset.write(qs, out)
        by = Counter((q["category"], q["split"]) for q in qs)
        print(f"{len(qs)} questions -> {out}")
        for cat in dict.fromkeys(q["category"] for q in qs):
            print(f"  {cat:13} dev {by[(cat, 'dev')]:2}  test {by[(cat, 'test')]:2}")
        print(f"  plain-language: {sum(q['style'] == 'plain' for q in qs)}")
    elif args.cmd == "index":
        from diligence_agent.indexer import build_index

        build_index(cfg, runner)
    elif args.cmd == "index-baseline":
        from diligence_agent import baseline

        baseline.build_index(cfg, runner)
    elif args.cmd == "eval":
        from diligence_agent import evaluate, tracing

        tracing.setup(cfg, "diligence-agent-eval")
        evaluate.run(
            args.system, args.split, cfg, runner, fake=args.fake, limit=args.limit, ids=args.ids, repeat=args.repeat
        )
    elif args.cmd == "label-sheet":
        from pathlib import Path

        from diligence_agent import calibrate, evalset

        qs = {q["id"]: q for q in evalset.load(cfg["eval"]["questions"])}
        rows = [r for rs in calibrate.load_answers(args.split).values() for r in rs]
        cal_sheet = Path(cfg["eval"]["human_labels"])
        if args.split == "dev":
            picked = calibrate.sample(rows, qs, n=cfg["preregistered"]["judge"]["labels"])
            out, guide, text = cal_sheet, cal_sheet.with_name("LABELING.md"), ""
        else:  # the judge failed calibration: hand labels grade the judge-checked test answers
            picked = calibrate.judged_unlabeled(rows, qs, set(calibrate.hand_labels(cal_sheet.parent)))
            out = cal_sheet.with_name("human_labels_test.csv")
            guide, text = cal_sheet.with_name("LABELING_test.md"), calibrate.LABEL_GUIDE
        calibrate.write_sheet(picked, qs, out, guide, guide=text)
        print(f"{len(picked)} answers -> {out} (guide: {guide})")
    elif args.cmd == "calibrate":
        from pathlib import Path

        from diligence_agent import calibrate

        kmin = cfg["preregistered"]["judge"]["kappa_min"]
        res = calibrate.calibrate(Path(cfg["eval"]["human_labels"]), calibrate.load_answers("dev"), kmin)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/judge_calibration.json").write_text(json.dumps(res, indent=1), encoding="utf8")
        print(json.dumps(res, indent=1))
    elif args.cmd == "regrade":
        from diligence_agent import evalset, evaluate

        path = evaluate.answers_path(args.system, args.split, repeat=args.repeat)
        qs = {q["id"]: q for q in evalset.load(cfg["eval"]["questions"])}
        rows = evaluate.regrade(evaluate.load_rows(path), qs, labels=evaluate.load_labels())
        evaluate.write_rows(rows, path)
        print(json.dumps(evaluate.headline(evaluate.summarize(rows, cfg)), indent=1))
    elif args.cmd == "judge":
        from diligence_agent import evalset, evaluate, grading, tracing

        path = evaluate.answers_path(args.system, args.split, repeat=args.repeat)
        rows = evaluate.load_rows(path)
        qs = {q["id"]: q for q in evalset.load(cfg["eval"]["questions"])}
        chat = grading.fake_judge if args.fake else grading.OllamaJudge(cfg["ollama_url"], cfg["judge_model"])
        rows = evaluate.judge_rows(rows, qs, chat, faithfulness=args.faithfulness, labels=evaluate.load_labels())
        evaluate.write_rows(rows, path)
        print(json.dumps(evaluate.headline(evaluate.summarize(rows, cfg)), indent=1))
    elif args.cmd == "figures":
        from diligence_agent import figures

        for path in figures.build(cfg):
            print(path)
    elif args.cmd == "analyze":
        from pathlib import Path

        from diligence_agent import analyze

        res = analyze.analyze(cfg, args.split)
        out = Path("reports") / ("eval.json" if args.split == "test" else "eval_dev.json")
        out.write_text(json.dumps(res, indent=1), encoding="utf8", newline="\n")
        print(json.dumps(analyze.brief(res), indent=1))
        print(f"-> {out}")
    elif args.cmd == "tools":
        print(json.dumps(runner.schemas(), indent=1))
    elif args.cmd == "tool":
        out = runner.call(args.name, args.args)
        print(
            json.dumps(
                {
                    "ok": out.ok,
                    "error": out.error,
                    "cache_hit": out.cache_hit,
                    "latency_ms": round(out.latency_ms),
                    "records": len(out.evidence),
                }
            )
        )
        print(out.as_message())
        return 0 if out.ok else 1
    elif args.cmd == "ask":
        from diligence_agent import agent, tracing

        tracing.setup(cfg, "diligence-agent-ask")
        prompts = {
            "planner": agent.PLANNER,
            "researcher": agent.RESEARCHER,
            "writer": agent.WRITER,
            "verifier": agent.VERIFIER,
        }
        result = agent.run_question(args.question, runner, cfg, tags=tracing.lineage(cfg, prompts, fake=args.fake))
        print(result.answer)
        print(json.dumps(result.summary() | {"plan": result.plan}, indent=1), file=sys.stderr)
    elif args.cmd == "lc":
        from diligence_agent import tracing
        from diligence_agent.lc_agent import SYSTEM_PROMPT, run_question

        tracing.setup(cfg, "diligence-agent-ask")
        tags = tracing.lineage(cfg, {"lc_agent": SYSTEM_PROMPT}, fake=args.fake)
        result = run_question(args.question, runner, cfg, tags=tags)
        print(result.answer)
        print(json.dumps(result.summary(), indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
