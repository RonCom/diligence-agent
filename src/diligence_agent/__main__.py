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
    t = sub.add_parser("tool", help="call one tool")
    t.add_argument("name")
    t.add_argument("args", nargs="?", default="{}", help="JSON arguments")
    lc = sub.add_parser("lc", help="answer a question with the LangChain tool-calling agent")
    lc.add_argument("question")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # labels contain characters such as "≥" that cp1252 can't print
        stream.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    from diligence_agent.tools import build_runner

    runner = build_runner(cfg, fake=args.fake)
    if args.cmd == "tools":
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
        from diligence_agent.lc_agent import run_question

        result = run_question(args.question, runner, cfg)
        print(result.answer)
        print(json.dumps(result.summary(), indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
