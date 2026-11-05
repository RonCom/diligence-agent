"""Load config.yaml and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    load_dotenv()
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f)


def sec_user_agent() -> str | None:
    """SEC rejects requests without a declared name and email (HTTP 403). Read it from .env, never from code."""
    value = os.environ.get("SEC_USER_AGENT", "").strip()
    return value if "@" in value else None
