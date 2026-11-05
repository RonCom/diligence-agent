"""Tool-result cache keyed by a hash of the tool name and its normalized arguments.

The same normalization feeds loop detection: two calls that differ only in case or spacing count as one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items()) if v is not None}
    if isinstance(value, list | tuple):
        return [normalize(v) for v in value]
    return value


def cache_key(tool: str, args: dict[str, Any], version: int = 1) -> str:
    """version is the tool's output-format version: bumping it retires cached results in the old format."""
    blob = json.dumps(
        {"tool": tool, "version": version, "args": normalize(args)}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class ToolCache:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> list[dict] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf8"))

    def put(self, key: str, records: list[dict]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(records), encoding="utf8")
        tmp.replace(path)
