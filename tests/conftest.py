from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from diligence_agent.tools import build_runner


@pytest.fixture(autouse=True, scope="session")
def no_tracing():
    """Tests write no traces; test_tracing.py turns tracing on against a temporary store."""
    mlflow.tracing.disable()
    yield


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    return {"data_dir": str(tmp_path / "data"), "tool_timeout_s": 10, "http": {"rate_limits": {}}}


@pytest.fixture
def runner(cfg):
    return build_runner(cfg, fake=True)
