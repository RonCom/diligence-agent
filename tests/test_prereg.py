from __future__ import annotations

from pathlib import Path

import yaml

from diligence_agent.prereg import can_score, validate

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def cfg_with(**changes) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    cfg["preregistered"] |= changes
    return cfg


def test_committed_block_is_valid():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    assert validate(cfg) == []
    assert [t["id"] for t in cfg["preregistered"]["tests"]] == ["H1", "H2", "H3", "H4"]


def test_dev_always_scorable():
    assert can_score(cfg_with(status="draft"), "dev", tags=set())[0]


def test_test_split_needs_frozen_tag():
    ok, why = can_score(cfg_with(status="draft"), "test", tags={"eval-test-v1"})
    assert not ok and "draft" in why
    frozen = cfg_with(status="frozen", approved="2026-11-05")
    assert not can_score(frozen, "test", tags=set())[0]
    assert can_score(frozen, "test", tags={"eval-test-v1"}) == (True, "eval-test-v1")


def test_frozen_needs_approval_date():
    assert "approval date" in validate(cfg_with(status="frozen", approved=None))[0]
