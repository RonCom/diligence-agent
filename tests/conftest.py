from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from diligence_agent.tools import build_runner


@pytest.fixture(autouse=True, scope="session")
def no_tracing(tmp_path_factory):
    """Tests write no traces and no runs to the repo; test_tracing.py turns tracing on against its own store."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path_factory.mktemp('mlflow') / 'mlflow.db').as_posix()}")
    mlflow.tracing.disable()
    yield


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    return {"data_dir": str(tmp_path / "data"), "tool_timeout_s": 10, "http": {"rate_limits": {}}}


@pytest.fixture
def runner(cfg):
    return build_runner(cfg, fake=True)
