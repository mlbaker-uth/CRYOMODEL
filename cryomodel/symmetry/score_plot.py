"""Plots for symmetry score quality across n within family."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_family_score_plot(phase_json: Path, *, family: Literal["c", "d"], out_png: Path | None = None) -> Path:
    phase_json = Path(phase_json).expanduser().resolve()
    with open(phase_json, encoding="utf-8") as fh:
        p = json.load(fh)
    cands = p.get("candidates") or []
    if family == "c":
        key = "cn_scores"
        title = "C_n quality (best across axis candidates)"
        ylab = "Pearson r"
    else:
        key = "dn_scores"
        title = "D_n quality (best across axis candidates)"
        ylab = "D_n score"
    best_by_n: dict[int, float] = {}
    for c in cands:
        scores = c.get(key) or {}
        for nk, sv in scores.items():
            n = int(nk)
            s = float(sv)
            if (n not in best_by_n) or (s > best_by_n[n]):
                best_by_n[n] = s
    if not best_by_n:
        raise ValueError(f"No {key} values in {phase_json}")
    ns = sorted(best_by_n)
    ys = [best_by_n[n] for n in ns]
    out = Path(out_png).expanduser().resolve() if out_png else phase_json.with_name(
        "symmetry_scores_Cn.png" if family == "c" else "symmetry_scores_Dn.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    ax.plot(ns, ys, marker="o", lw=1.8)
    ax.set_xlabel("n")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out

