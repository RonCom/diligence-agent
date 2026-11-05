from __future__ import annotations

from pathlib import Path

import pytest

from diligence_agent.tools import build_runner


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    return {"data_dir": str(tmp_path / "data"), "tool_timeout_s": 10, "http": {"rate_limits": {}}}


@pytest.fixture
def runner(cfg):
    return build_runner(cfg, fake=True)
