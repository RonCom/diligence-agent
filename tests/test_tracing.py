"""MLflow traces from the LangChain agent: one root span, a span per model call and per tool call."""

from __future__ import annotations

import mlflow
import pytest
from langchain_core.messages import AIMessage

from diligence_agent.lc_agent import PROMPT_SHA, run_question
from diligence_agent.tracing import cost, lineage
from tests.test_lc_agent import call, scripted


@pytest.fixture
def tracing(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("test")
    mlflow.tracing.enable()
    yield
    mlflow.set_tracking_uri(previous)  # resets the tracer, so disable after it
    mlflow.tracing.disable()


def usage(i: int, o: int) -> dict:
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def test_trace_has_step_and_tool_spans(tracing, runner, cfg):
    first = call("orange_book", {"drug": "Trikafta"}, 1)
    first.usage_metadata = usage(1000, 20)
    final = AIMessage(
        content="Expires 2033-07-15 [orange_book:N212273/patent/9012496].", usage_metadata=usage(3000, 40)
    )
    cfg = {**cfg, "chat_model": "m", "pricing": {"m": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}}}
    r = run_question("When?", runner, cfg, model=scripted(first, final), tags={"git_commit": "abc"})
    assert r.trace_id and r.trace_id.startswith("tr-")
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(r.trace_id)
    spans = {s.name: s for s in trace.data.spans}
    assert {"lc_agent", "step_1", "tool:orange_book", "step_2"} <= set(spans)

    step = spans["step_1"].attributes
    assert step["mlflow.chat.tokenUsage"]["input_tokens"] == 1000
    assert step["mlflow.llm.model"] == "m" and step["prompt_sha"] == PROMPT_SHA
    assert step["mlflow.llm.cost"]["total_cost"] == pytest.approx(1000 / 1e6 + 20 * 5 / 1e6)

    tool = spans["tool:orange_book"].attributes
    assert tool["ok"] is True and tool["cache_hit"] is False and tool["records"] > 0
    assert "orange_book:N212273/patent/9012496" in tool["record_keys"]

    root = spans["lc_agent"].attributes
    assert root["input_tokens"] == 4000 and root["output_tokens"] == 60 and root["tool_calls"] == 1
    assert root["cost_usd"] == pytest.approx(4000 / 1e6 + 60 * 5 / 1e6)
    assert trace.info.tags["git_commit"] == "abc" and trace.info.tags["agent"] == "langchain"


def test_repeat_call_span_marked(tracing, runner, cfg):
    model = scripted(
        call("fda_label", {"drug": "Trikafta"}, 1), call("fda_label", {"drug": "Trikafta"}, 2), AIMessage(content="x")
    )
    r = run_question("q", runner, cfg, model=model)
    mlflow.flush_trace_async_logging()
    tools = [s for s in mlflow.get_trace(r.trace_id).data.spans if s.name == "tool:fda_label"]
    assert [s.attributes.get("repeated", False) for s in tools] == [False, True]


def test_no_trace_id_when_tracing_off(runner, cfg):
    r = run_question("q", runner, cfg, model=scripted(AIMessage(content="NOT IN SOURCES")))
    assert r.trace_id is None and r.declined


def test_cost_table():
    assert cost({}, "gemma4:e4b", 10_000, 500)["total_cost"] == 0.0
    c = cost({"x": {"input_per_mtok": 3, "output_per_mtok": 15}}, "x", 1_000_000, 100_000)
    assert c == {"input_cost": 3.0, "output_cost": 1.5, "total_cost": 4.5}


def test_lineage_fake_skips_ollama(cfg):
    lin = lineage(cfg, {"answer": "prompt text"}, fake=True)
    assert lin["fake"] is True and len(lin["answer_prompt_sha"]) == 12
    assert "ollama_version" not in lin and "git_commit" in lin
