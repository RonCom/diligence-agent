"""MLflow tracing and lineage, following policy-rag's tracking.py and answer.py.

One trace per request. Inside it, one span per model call (an agent step) and one per tool call. Model spans carry
the model id, input and output tokens (mlflow.chat.tokenUsage, which MLflow sums per trace), USD from the pricing
table in config.yaml (mlflow.llm.cost; local models are absent from it and cost $0), latency and Ollama's own
timings. Tool spans carry the arguments, latency, cache hit, record count and any error. Runs log lineage: git
commit, prompt hashes, Ollama version, model digests and the search index build, because policy-rag showed that
the same prompts at temperature 0 give different answers after a model re-pull.

Local SQLite store (mlflow.db), no tracking server: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

DEFAULT_URI = "sqlite:///mlflow.db"


def setup(cfg: dict, experiment: str) -> None:
    mlflow.set_tracking_uri(cfg.get("mlflow_uri", DEFAULT_URI))
    mlflow.set_experiment(experiment)


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:12]


def cost(pricing: dict | None, model: str, input_tokens: int, output_tokens: int) -> dict[str, float]:
    """USD from a per-model price table in USD per million tokens; a model absent from it costs 0."""
    p = (pricing or {}).get(model) or {}
    i = input_tokens * float(p.get("input_per_mtok", 0)) / 1e6
    o = output_tokens * float(p.get("output_per_mtok", 0)) / 1e6
    return {"input_cost": i, "output_cost": o, "total_cost": i + o}


def ollama_runtime(url: str, models: dict[str, str]) -> dict[str, str]:
    """Ollama version and the digest of each model; 'unavailable' when no server answers (tests, --fake)."""
    url = url.rstrip("/")
    try:
        version = httpx.get(f"{url}/api/version", timeout=5).json().get("version", "unknown")
        tags = {m["name"]: m.get("digest", "")[:12] for m in httpx.get(f"{url}/api/tags", timeout=5).json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError):
        return {"ollama_version": "unavailable"}

    def digest(name: str) -> str:
        return tags.get(name) or tags.get(f"{name}:latest") or "missing"

    return {"ollama_version": version} | {f"{role}_digest": digest(name) for role, name in models.items()}


def git_commit(cwd: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "none"
    commit = out.stdout.strip() or "none"
    return f"{commit}-dirty" if dirty and commit != "none" else commit


def lineage(cfg: dict, prompts: dict[str, str], fake: bool = False) -> dict[str, Any]:
    """What a metric depends on besides the code: prompts, models, the drug set and the search index."""
    ds = cfg.get("docsearch", {})
    out: dict[str, Any] = {"git_commit": git_commit(), "fake": fake, "chat_model": cfg.get("chat_model")}
    out |= {f"{name}_prompt_sha": sha(text) for name, text in prompts.items()}
    products = Path(ds.get("products", "eval/drug_set.yaml"))
    if products.exists():
        out["drug_set_sha"] = sha(products.read_bytes())
    meta = Path(cfg.get("data_dir", "data")) / "index" / "meta.json"
    if meta.exists() and not fake:
        m = json.loads(meta.read_text(encoding="utf8"))
        out |= {"index_built_at": m.get("built_at"), "index_chunks": m.get("chunks")}
    if not fake:
        models = {
            "chat_model": cfg.get("chat_model", ""),
            "judge_model": cfg.get("judge_model", ""),
            "embed_model": ds.get("embed_model", ""),
        }
        out |= ollama_runtime(cfg.get("ollama_url", "http://localhost:11434"), models)
    return out


def model_span_attributes(
    model: str, usage: dict, pricing: dict | None, latency_ms: float, prompt_sha: str, extra: dict | None = None
) -> dict[str, Any]:
    i, o = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return {
        SpanAttributeKey.MODEL: model,
        SpanAttributeKey.CHAT_USAGE: {"input_tokens": i, "output_tokens": o, "total_tokens": i + o},
        SpanAttributeKey.LLM_COST: cost(pricing, model, i, o),
        "latency_ms": latency_ms,
        "prompt_sha": prompt_sha,
    } | (extra or {})


def ollama_timings(response_metadata: dict) -> dict[str, float]:
    """Ollama reports durations in nanoseconds; keep load, prompt processing and generation in ms."""
    keys = {
        "load_duration": "load_ms",
        "prompt_eval_duration": "prompt_eval_ms",
        "eval_duration": "eval_ms",
        "total_duration": "ollama_total_ms",
    }
    return {f"ollama.{v}": float(response_metadata[k]) / 1e6 for k, v in keys.items() if response_metadata.get(k)}


__all__ = ["SpanType", "cost", "lineage", "model_span_attributes", "ollama_timings", "setup", "sha"]
