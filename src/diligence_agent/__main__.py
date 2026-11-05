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
    lc = sub.add_parser("lc", help="answer a question with the LangChain tool-calling agent")
    lc.add_argument("question")
    be = sub.add_parser("build-eval", help="compute gold answers from the sources and write the eval set")
    be.add_argument("--out", default="eval/questions.jsonl")
    be.add_argument("--check", action="store_true", help="recompute gold and list changes; write nothing")
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
