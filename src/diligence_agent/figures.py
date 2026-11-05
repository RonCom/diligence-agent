"""README figures, built from saved reports (no model calls): `diligence-agent figures`.

  accuracy_by_category.png  correct_strict per question category, test repeat 1   -> where each system loses points
  tokens_vs_accuracy.png    tokens per question vs correct_strict with 95% CI      -> what the extra calls buy
  latency.png               p50/p95 latency per system; the custom agent by role   -> where the time goes

Colors follow fixed categorical slots (blue, orange, aqua), validated for color-vision deficiency.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from diligence_agent import evalset, evaluate

LABEL = {"agent": "Custom agent", "langchain": "LangChain agent", "baseline": "Single-call RAG"}
COLOR = {"agent": "#2a78d6", "langchain": "#eb6834", "baseline": "#1baf7a"}
ORDER = ("agent", "langchain", "baseline")
CATEGORY = {"patent": "Patent expiry", "exclusivity": "Exclusivity expiry", "approval": "First approval date",
            "indication": "Approved for X?", "trial": "Trial status", "readouts": "Phase 3 readouts, 12 months",
            "tenk": "10-K royalty terms, sales", "conflict": "10-K vs Orange Book",
            "biologic": "Biologics (no patents)",
            "out_of_scope": "Out of scope"}  # fmt: skip
INK, MUTED, GRID, SURFACE = "#2b2a27", "#6f6e69", "#e4e3df", "#fcfcfb"
ROLES = ("planner", "researcher", "writer", "tools", "verifier")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})  # fmt: skip
    return plt


def accuracy_by_category(res: dict, n: dict[str, int], out: Path, plt) -> None:
    cats = [c for c in CATEGORY if c in res["systems"]["agent"]["by_category"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    h = 0.26
    for i, s in enumerate(ORDER):
        vals = [100 * res["systems"][s]["by_category"][c]["mean"] for c in cats]
        y = [k + (i - 1) * h for k in range(len(cats))]
        ax.barh(y, vals, h * 0.85, color=COLOR[s], label=LABEL[s])
        for yy, v in zip(y, vals, strict=True):
            ax.text(v + 1.2, yy, f"{v:.0f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(cats)), [f"{CATEGORY[c]} (n = {n[c]})" for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("% correct (strict), test split, 62 questions")
    ax.grid(axis="y", visible=False)
    ax.set_title("Both agents gain most on trial status, patents and approval dates")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.04), fontsize=8)
    fig.savefig(out / "accuracy_by_category.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def tokens_vs_accuracy(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for s in ORDER:
        m = res["systems"][s]
        c = m["correct_strict"]
        x, y = m["tokens_mean"], 100 * c["mean"]
        ax.errorbar(x, y, yerr=[[y - 100 * c["lo"]], [100 * c["hi"] - y]], fmt="o", ms=8, color=COLOR[s],
                    ecolor=COLOR[s], elinewidth=2, capsize=0, mec=SURFACE, mew=2)  # fmt: skip
        ax.annotate(f"{LABEL[s]}\n{m['model_calls_mean']:.1f} model calls, p50 {m['latency_p50_s']:.1f} s",
                    (x, y), xytext=(10, -4), textcoords="offset points", fontsize=7.5, color=INK,
                    va="center")  # fmt: skip
    ax.set_xlim(0, 11_000)
    ax.set_ylim(40, 100)
    ax.set_xlabel("Tokens per question (input + output, mean)")
    ax.set_ylabel("% correct (strict), 95% CI")
    ax.set_title("Local models cost $0, so tokens and seconds are the price")
    fig.savefig(out / "tokens_vs_accuracy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def latency(res: dict, agent_rows: list[dict], out: Path, plt) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 2.9), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})
    for k, s in enumerate(ORDER):
        m = res["systems"][s]
        a1.plot([m["latency_p50_s"], m["latency_p95_s"]], [k, k], color=COLOR[s], lw=2)
        a1.plot(m["latency_p50_s"], k, "o", ms=8, color=COLOR[s], mec=SURFACE, mew=2)
        a1.plot(m["latency_p95_s"], k, "o", ms=8, color=SURFACE, mec=COLOR[s], mew=2)
        a1.text(m["latency_p95_s"] + 0.9, k, f"{m['latency_p50_s']:.1f} / {m['latency_p95_s']:.1f} s", va="center",
                fontsize=7, color=MUTED)  # fmt: skip
    a1.set_yticks(range(3), [LABEL[s] for s in ORDER])
    a1.invert_yaxis()
    a1.set_xlim(0, 19)
    a1.set_ylim(2.6, -0.6)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("Seconds per question (filled p50, open p95)")
    a1.set_title("Latency per question")
    means = {r: statistics.mean((x.get("role_ms") or {}).get(r, 0) for x in agent_rows) / 1000 for r in ROLES}
    roles = sorted(ROLES, key=lambda r: -means[r])
    a2.barh(range(len(roles)), [means[r] for r in roles], 0.6, color=COLOR["agent"])
    for k, r in enumerate(roles):
        a2.text(means[r] + 0.06, k, f"{means[r]:.1f} s", va="center", fontsize=7, color=MUTED)
    a2.set_yticks(range(len(roles)), [r.capitalize() for r in roles])
    a2.invert_yaxis()
    a2.set_xlim(0, 4)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("Mean seconds per question")
    a2.set_title(f"Custom agent by role (total {sum(means.values()):.1f} s)")
    fig.savefig(out / "latency.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def build(cfg: dict, reports: str | Path = "reports") -> list[Path]:
    reports = Path(reports)
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((reports / "eval.json").read_text(encoding="utf8"))
    qs = [q for q in evalset.load(cfg["eval"]["questions"]) if q["split"] == "test"]
    n = {c: sum(q["category"] == c for q in qs) for c in CATEGORY}
    agent_rows = evaluate.load_rows(evaluate.answers_path("agent", "test", reports, 1))
    plt = _plt()
    accuracy_by_category(res, n, out, plt)
    tokens_vs_accuracy(res, out, plt)
    latency(res, agent_rows, out, plt)
    return sorted(out.glob("*.png"))
