"""Bootstrap intervals for a mean and for a paired difference between systems on the same questions."""

from __future__ import annotations

import numpy as np


def mean_ci(values, resamples: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    a = (1 - ci) / 2
    return {"n": len(v), "mean": float(v.mean()), "lo": float(np.quantile(means, a)),
            "hi": float(np.quantile(means, 1 - a))}  # fmt: skip


def paired_diff_ci(a: dict, b: dict, resamples: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    """a, b: question id -> score. Uses the questions both systems have a score for."""
    ids = sorted(k for k in a.keys() & b.keys() if a[k] is not None and b[k] is not None)
    d = np.array([a[k] - b[k] for k in ids], dtype=float)
    out = mean_ci(d, resamples, seed, ci)
    return out | {"above_zero": out["lo"] is not None and out["lo"] > 0,
                  "excludes_zero": out["lo"] is not None and (out["lo"] > 0 or out["hi"] < 0)}  # fmt: skip
