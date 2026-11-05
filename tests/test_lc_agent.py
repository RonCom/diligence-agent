"""The LangChain agent with a scripted chat model: no Ollama needed."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from diligence_agent.lc_agent import run_question


class ScriptedToolModel(GenericFakeChatModel):
    """GenericFakeChatModel replays messages; bind_tools is a no-op so create_agent accepts it."""

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def scripted(*messages: AIMessage) -> ScriptedToolModel:
    return ScriptedToolModel(messages=iter(messages))


def call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def test_answers_with_citations(runner, cfg):
    model = scripted(
        call("orange_book", {"drug": "Trikafta"}, 1),
        AIMessage(content="Patent 9012496 expires 2033-07-15 [orange_book:N212273/patent/9012496]."),
    )
    r = run_question("When does patent 9012496 expire?", runner, cfg, model=model)
    s = r.summary()
    assert not r.declined
    assert s["citations"] == 1 and s["citations_to_unseen_records"] == []
    assert s["tool_calls"][0]["ok"] and s["tool_calls"][0]["records"] > 0
    assert s["model_calls"] == 2


def test_bad_args_go_back_to_model_then_repaired(runner, cfg):
    model = scripted(
        call("ctgov_search", {"phase": "PHASE3"}, 1),  # missing intervention
        call("ctgov_search", {"intervention": "elexacaftor", "phase": "Phase 3"}, 2),
        AIMessage(content="NCT03691779 is a completed Phase 3 trial [ctgov:NCT03691779]."),
    )
    r = run_question("Phase 3 trials of elexacaftor?", runner, cfg, model=model)
    s = r.summary()
    assert [c["ok"] for c in s["tool_calls"]] == [False, True]
    assert s["tool_call_validity"] == 0.5
    assert "invalid arguments" in s["tool_calls"][0]["error"]


def test_repeat_call_blocked_and_unseen_citation_flagged(runner, cfg):
    model = scripted(
        call("fda_label", {"drug": "Trikafta"}, 1),
        call("fda_label", {"drug": "  trikafta "}, 2),
        AIMessage(content="It is approved for CF [fda_label:made-up-id]."),
    )
    r = run_question("Is Trikafta approved for CF?", runner, cfg, model=model)
    s = r.summary()
    assert s["repeated_calls"] == 1 and len(s["tool_calls"]) == 1
    assert s["citations_to_unseen_records"] == ["fda_label:made-up-id"]


def test_step_budget_ends_with_decline(runner, cfg):
    cfg = {**cfg, "budgets": {"max_steps": 2}}
    model = scripted(*[call("pubmed_search", {"query": f"elexacaftor {i}"}, i) for i in range(5)])
    r = run_question("Anything?", runner, cfg, model=model)
    assert r.declined and "step budget" in r.answer
    assert r.model_calls == 2 and r.budget_stop


def test_decline_detected(runner, cfg):
    model = scripted(call("orange_book", {"drug": "Tremfya"}, 1), AIMessage(content="NOT IN SOURCES"))
    r = run_question("When does Tremfya's last Orange Book patent expire?", runner, cfg, model=model)
    assert r.declined and r.summary()["tool_calls"][0]["records"] == 0
