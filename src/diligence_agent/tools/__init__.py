"""Read-only tools the agents can call."""

from __future__ import annotations

import os
from pathlib import Path

from diligence_agent.cache import ToolCache
from diligence_agent.http import HttpClient
from diligence_agent.tools.base import Tool, ToolContext, ToolOutcome, ToolRunner
from diligence_agent.tools.ctgov import CtgovTool
from diligence_agent.tools.docsearch import DocSearchTool
from diligence_agent.tools.edgar import EdgarFilingsTool
from diligence_agent.tools.openfda import DrugsAtFdaTool, LabelTool
from diligence_agent.tools.orange_book import OrangeBookTool
from diligence_agent.tools.pubmed import PubmedTool

__all__ = ["ALL_TOOLS", "Tool", "ToolContext", "ToolOutcome", "ToolRunner", "build_runner"]

ALL_TOOLS: list[type[Tool]] = [
    OrangeBookTool,
    CtgovTool,
    LabelTool,
    DrugsAtFdaTool,
    EdgarFilingsTool,
    PubmedTool,
    DocSearchTool,
]


def build_runner(cfg: dict, fake: bool = False, use_cache: bool = True) -> ToolRunner:
    """Live mode caches results under data/cache/tools. Fake mode serves recorded responses and caches nothing."""
    data_dir = Path(cfg.get("data_dir", "data"))
    transport = None
    if fake:
        from diligence_agent.fake import fake_transport

        transport = fake_transport()
    http = HttpClient.from_config(cfg, transport=transport, **({"sleep": lambda s: None} if fake else {}))
    env = {k: os.environ[k] for k in ("SEC_USER_AGENT", "OPENFDA_API_KEY") if os.environ.get(k)}
    ctx = ToolContext(http=http, data_dir=data_dir, fake=fake, env=env, settings=cfg)
    cache = ToolCache(data_dir / "cache" / "tools") if use_cache and not fake else None
    return ToolRunner([t() for t in ALL_TOOLS], ctx, cache=cache, timeout_s=cfg.get("tool_timeout_s", 60))
