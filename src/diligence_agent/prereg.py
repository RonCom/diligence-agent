"""Guard for the pre-registered tests in config.yaml: the test split is scored once, from a frozen tag."""

from __future__ import annotations

import subprocess

STATUSES = ("draft", "approved", "frozen")


def validate(cfg: dict) -> list[str]:
    """Problems with the preregistered block; empty when it is well formed."""
    p = cfg.get("preregistered") or {}
    problems = []
    if p.get("status") not in STATUSES:
        problems.append(f"status must be one of {STATUSES}")
    if p.get("status") in ("approved", "frozen") and not p.get("approved"):
        problems.append("an approved or frozen block needs its approval date")
    ids = [t.get("id") for t in p.get("tests", [])]
    if len(ids) != len(set(ids)) or not ids:
        problems.append("tests need unique ids")
    for t in p.get("tests", []):
        if not t.get("pass_if"):
            problems.append(f"{t.get('id')}: pass_if missing")
    if not 0 < float(p.get("judge", {}).get("kappa_min", 0)) <= 1:
        problems.append("judge.kappa_min must be in (0, 1]")
    return problems


def tags_at_head(cwd: str = ".") -> set[str]:
    try:
        out = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        ).stdout
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return set()
    return set() if dirty else set(out.split())


def can_score(cfg: dict, split: str, tags: set[str] | None = None) -> tuple[bool, str]:
    """Dev can always be scored. Test needs a frozen block and a clean checkout at the frozen tag."""
    if split != "test":
        return True, "dev"
    if problems := validate(cfg):
        return False, "; ".join(problems)
    p = cfg["preregistered"]
    if p["status"] != "frozen":
        return False, f"pre-registered tests are {p['status']}, not frozen"
    tags = tags_at_head() if tags is None else tags
    if p["test_tag"] not in tags:
        return False, f"HEAD is not a clean checkout of tag {p['test_tag']}"
    return True, p["test_tag"]
