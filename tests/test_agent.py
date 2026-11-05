"""The custom agent with a scripted chat model and fake tools: no Ollama needed."""

from __future__ import annotations

import json

import mlflow

from diligence_agent.agent import Agent, Budget, split_claims, values_in_records
from diligence_agent.llm import Reply, ScriptedChat

PATENT = "orange_book:N212273/patent/9012496"


def plan(*steps, oos=False):
    return Reply(content=json.dumps({"out_of_scope": oos, "steps": [{"goal": g, "tool": t} for g, t in steps]}))


def call(name, args):
    return Reply(tool_calls=[{"name": name, "arguments": args}], input_tokens=100, output_tokens=10)


def verdicts(*supported):
    return Reply(content=json.dumps({"results": [{"claim": i + 1, "supported": s} for i, s in enumerate(supported)]}))


def agent(runner, cfg, *replies, **budgets):
    cfg = cfg | {"budgets": {"max_steps": 12, "max_tokens": 60000, **budgets}}
    return Agent(runner, ScriptedChat(list(replies)), cfg)


def test_plan_research_write_verify(runner, cfg):
    a = agent(runner, cfg,
              plan(("Find Trikafta's Orange Book patents", "orange_book")),
              call("orange_book", {"drug": "Trikafta"}),
              Reply(content="Found the patents."),
              Reply(content=f"Patent 9012496 expires 2033-07-15 [{PATENT}]."),
              verdicts(True))  # fmt: skip
    r = a.run("When does patent 9012496 expire?")
    s = r.summary()
    assert r.answer == f"Patent 9012496 expires 2033-07-15 [{PATENT}]."
    assert s["claims"] == 1 and s["claims_supported"] == 1 and s["claims_cut"] == 0
    assert s["model_calls"] == 5 and s["tool_calls"][0]["ok"]
    assert s["citations_to_unseen_records"] == [] and set(s["role_ms"]) >= {"planner", "researcher", "writer"}


def test_value_not_in_record_is_cut_after_second_round(runner, cfg):
    wrong = f"Patent 9012496 expires 2040-01-01 [{PATENT}]."
    a = agent(
        runner,
        cfg,
        plan(("patents", "orange_book")),
        call("orange_book", {"drug": "Trikafta"}),
        Reply(content="done"),
        Reply(content=wrong),  # draft: date not in the record -> code check fails
        Reply(content="nothing new"),  # second research round finds nothing
        Reply(content=wrong),
    )  # redraft repeats it; verifier never called (code fails)
    r = a.run("When does patent 9012496 expire?")  # fmt: skip
    assert r.research_rounds == 2 and r.claims_cut == 1
    assert r.answer == "NOT IN SOURCES"
    assert "value not in it" in r.claims[0].reason


def test_uncited_sentence_cut_keeps_supported_claim(runner, cfg):
    a = agent(runner, cfg,
              plan(("patents", "orange_book")),
              call("orange_book", {"drug": "Trikafta"}), Reply(content="done"),
              Reply(content=f"Patent 9012496 expires 2033-07-15 [{PATENT}]. Trikafta is a great drug."),
              verdicts(True),
              Reply(content="nothing"),
              Reply(content=f"Patent 9012496 expires 2033-07-15 [{PATENT}]. Trikafta is a great drug."),
              verdicts(True))  # fmt: skip
    r = a.run("q?")
    assert r.answer == f"Patent 9012496 expires 2033-07-15 [{PATENT}]." and r.claims_cut == 1


def test_out_of_scope_flag_is_logged_and_writer_decides(runner, cfg):
    r = agent(runner, cfg, plan(("price", "doc_search"), oos=True),
              call("doc_search", {"query": "Trikafta list price"}), Reply(content="nothing on price"),
              Reply(content="NOT IN SOURCES")).run("What is Trikafta's list price?")  # fmt: skip
    assert r.out_of_scope and r.answer == "NOT IN SOURCES" and len(r.calls) == 1


def test_invalid_args_get_one_repair(runner, cfg):
    chat = ScriptedChat([
        plan(("trials", "ctgov_search")),
        call("ctgov_search", {"phase": "PHASE3"}),                       # invalid: nothing to search
        call("ctgov_search", {"intervention": "elexacaftor", "phase": "PHASE3"}),
        Reply(content="done"),
        Reply(content="NCT03691779 is a completed Phase 3 trial [ctgov:NCT03691779]."),
        verdicts(True),
    ])  # fmt: skip
    r = Agent(runner, chat, cfg).run("Phase 3 trials of elexacaftor?")
    assert r.repair_attempts == 1 and [c.ok for c in r.calls] == [False, True]
    tool_msgs = [m for m in chat.requests[2]["messages"] if m["role"] == "tool"]
    assert "Fix the arguments" in tool_msgs[0]["content"]


def test_repeat_call_refused(runner, cfg):
    r = agent(runner, cfg,
              plan(("patents", "orange_book")),
              call("orange_book", {"drug": "Trikafta"}), call("orange_book", {"drug": " trikafta "}),
              Reply(content="done"), Reply(content="NOT IN SOURCES")).run("q?")  # fmt: skip
    assert r.repeated_calls == 1 and len(r.calls) == 1


def test_loop_stop_after_idle_turns(runner, cfg):
    cfg = cfg | {"loop_detection": {"steps_without_new_evidence": 2}}
    r = agent(runner, cfg,
              plan(("a", "pubmed_search"), ("b", "pubmed_search")),
              call("pubmed_search", {"query": "zzz one"}), call("pubmed_search", {"query": "zzz two"}),
              Reply(content="NOT IN SOURCES")).run("q?")  # fmt: skip
    assert r.loop_stops == 1 and len(r.calls) == 2


def test_step_budget_leaves_room_for_writer(runner, cfg):
    replies = [plan(("a", "pubmed_search"))] + [call("pubmed_search", {"query": f"elexacaftor {i}"}) for i in range(9)]
    a = agent(runner, cfg, *replies, max_steps=6)
    r = a.run("q?")
    assert r.budget_stop == "steps" and r.model_calls <= 6


def test_claim_helpers():
    claims = split_claims(f"First claim 2033 [{PATENT}]. Second has no cite. 4% royalty [sec_filing:x/1].")
    assert [c.cited for c in claims] == [[PATENT], [], ["sec_filing:x/1"]]
    assert values_in_records("Expires July 15, 2033 at 4%.", ["expires 2033-07-15", "a 4% rate"])
    assert not values_in_records("Expires 2034.", ["expires 2033-07-15"])


def test_budget_accounting():
    b = Budget(max_steps=3, max_tokens=100)
    b.charge(Reply(input_tokens=60, output_tokens=50), 0.0)
    assert b.exhausted() == "tokens" and b.exhausted(reserve=2) == "steps"


def test_spans_per_role(runner, cfg, tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    mlflow.set_experiment("t")
    mlflow.tracing.enable()
    try:
        r = agent(
            runner,
            cfg,
            plan(("p", "orange_book")),
            call("orange_book", {"drug": "Trikafta"}),
            Reply(content="done"),
            Reply(content=f"Expires 2033-07-15 [{PATENT}]."),
            verdicts(True),
        ).run("q?")
        mlflow.flush_trace_async_logging()
        names = [s.name for s in mlflow.get_trace(r.trace_id).data.spans]
    finally:
        mlflow.set_tracking_uri(previous)
        mlflow.tracing.disable()  # fmt: skip
    assert names[0] == "agent" and {"planner", "tool:orange_book", "writer", "verifier"} <= set(names)
    assert any(n.startswith("researcher_") for n in names)


def test_citations_one_or_many_per_bracket():
    from diligence_agent.evidence import citations, strip_citations

    text = "Four trials [ctgov:NCT03903835, ctgov:NCT06629779]. Rate [sec_filing:x/1]; see [1] and [note]."
    assert citations(text) == ["ctgov:NCT03903835", "ctgov:NCT06629779", "sec_filing:x/1"]
    assert strip_citations(text) == "Four trials . Rate ; see [1] and [note]."
    claims = split_claims("Two trials end in 2027 [ctgov:NCT03903835, ctgov:NCT06629779].")
    assert claims[0].cited == ["ctgov:NCT03903835", "ctgov:NCT06629779"]
