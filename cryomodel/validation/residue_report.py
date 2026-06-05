# cryomodel/validation/residue_report.py
"""Human-readable per-residue validation reports (Coot / spreadsheet friendly)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .composite_score import resolve_bfactor_column


def _fmt_num(val: Any, *, nd: int = 2, suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "n/a"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return str(val)
    if nd == 0:
        return f"{int(round(x))}{suffix}"
    return f"{x:.{nd}f}{suffix}"


def _rama_label(row: pd.Series) -> str:
    cls = str(row.get("rama_class") or "").strip().lower()
    if cls in ("favored", "allowed", "outlier"):
        return cls
    if float(row.get("rama_outlier", 0) or 0) >= 0.5:
        return "outlier"
    pen = float(row.get("rama_penalty", 0) or 0)
    if pen <= 0.06:
        return "favored"
    if pen <= 1.5:
        return "allowed"
    return "outlier"


def _rotamer_label(row: pd.Series) -> str:
    prob = float(row.get("rotamer_prob", 0.5) or 0.5)
    deg = float(row.get("rotamer_chi1_nearest_deg", 0) or 0)
    if not np.isfinite(prob):
        return "n/a"
    if deg > 40 or prob < 0.25:
        return "outlier"
    if prob >= 0.6 and deg <= 25:
        return "favored"
    if prob >= 0.35:
        return "allowed"
    return "outlier"


def resolve_validate_score_column(df: pd.DataFrame, metric: str = "auto") -> str:
    """Column used as the headline validate score (default: same as B-factor badness)."""
    m = (metric or "auto").strip().lower()
    if m == "quality":
        return "composite_quality_0_100"
    if m == "badness":
        return "composite_badness_0_100"
    if m == "band":
        return "composite_band_deviation_0_100"
    return resolve_bfactor_column(df, "auto")


def build_residue_report_table(
    df: pd.DataFrame,
    *,
    score_column: Optional[str] = None,
) -> pd.DataFrame:
    """Structured report table with labels and key metrics."""
    if score_column is None:
        score_column = resolve_validate_score_column(df, "auto")
    if score_column not in df.columns:
        raise KeyError(f"Score column {score_column!r} not in dataframe")

    out = pd.DataFrame()
    out["chain"] = df["chain"].astype(str)
    out["resi"] = df["resi"]
    out["seqid"] = df["seqid"].astype(str)
    out["resname"] = df["resname"].astype(str)
    out["validate_score"] = df[score_column].astype(float)

    for col in (
        "composite_quality_0_100",
        "composite_badness_0_100",
        "composite_band_deviation_0_100",
        "composite_strictness",
    ):
        if col in df.columns:
            out[col] = df[col]

    out["rama"] = df.apply(_rama_label, axis=1)
    out["rotamer"] = df.apply(_rotamer_label, axis=1)

    metric_cols = [
        ("local_res", "local_res_A"),
        ("ringer_Z", "ringer_Z"),
        ("Q_mean", "Q_mean"),
        ("CC_mask", "CC_mask"),
        ("ZNCC", "ZNCC"),
        ("CC_half1", "CC_half1"),
        ("CC_half2", "CC_half2"),
        ("CC_delta", "CC_delta"),
        ("steric_clashes", "steric_clashes"),
        ("clashscore_z", "clashscore_z"),
        ("ramachandran_prob", "ramachandran_prob"),
        ("rotamer_prob", "rotamer_prob"),
        ("rotamer_chi1_nearest_deg", "rotamer_chi1_deg"),
        ("rama_penalty", "rama_penalty"),
        ("cb_deviation_A", "cb_deviation_A"),
        ("ringer_peak_deg", "ringer_peak_deg"),
        ("ringer_to_rotamer_deg", "ringer_to_rotamer_deg"),
        ("ringer_half_drop", "ringer_half_drop"),
        ("continuity_score", "ca_tube_continuity"),
    ]
    for src, dst in metric_cols:
        if src in df.columns:
            out[dst] = df[src]

    for col in df.columns:
        if col.endswith("_z_residual") and col not in out.columns:
            out[col] = df[col]

    if "bfactor_flagged" in df.columns:
        out["flagged"] = df["bfactor_flagged"].astype(int)

    return out


def format_residue_report_line(row: pd.Series, *, score_column: str = "validate_score") -> str:
    """One-line summary, e.g. for Coot annotation or grep."""
    resi = row.get("resi", row.get("seqid", "?"))
    resname = str(row.get("resname", "???")).strip()
    chain = str(row.get("chain", "?")).strip()
    score = row.get(score_column, row.get("validate_score", np.nan))

    parts = [
        f"{resi} {resname} {chain}",
        f"validate score: {_fmt_num(score, nd=1)}",
        f"rama: {row.get('rama', 'n/a')}",
        f"rotamer: {row.get('rotamer', 'n/a')}",
    ]

    if "ringer_Z" in row.index and pd.notna(row.get("ringer_Z")):
        parts.append(f"emRinger: {_fmt_num(row['ringer_Z'], nd=2)}")
    if "Q_mean" in row.index and pd.notna(row.get("Q_mean")):
        parts.append(f"Q-score: {_fmt_num(row['Q_mean'], nd=3)}")
    if "steric_clashes" in row.index and pd.notna(row.get("steric_clashes")):
        n = int(round(float(row["steric_clashes"])))
        parts.append(f"clash: {n} atom{'s' if n != 1 else ''}")
    if "CC_mask" in row.index and pd.notna(row.get("CC_mask")):
        parts.append(f"CC: {_fmt_num(row['CC_mask'], nd=2)}")
    if "local_res_A" in row.index and pd.notna(row.get("local_res_A")):
        parts.append(f"local res: {_fmt_num(row['local_res_A'], nd=1)} Å")
    if "composite_band_deviation_0_100" in row.index and pd.notna(row.get("composite_band_deviation_0_100")):
        parts.append(f"band dev: {_fmt_num(row['composite_band_deviation_0_100'], nd=1)}")
    if row.get("flagged") == 1:
        parts.append("FLAGGED")

    return ", ".join(parts)


def write_residue_report(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    score_column: Optional[str] = None,
    basename: str = "residue_report",
) -> tuple[Path, Path]:
    """Write ``{basename}.csv`` and ``{basename}.txt`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if score_column is None:
        score_column = resolve_validate_score_column(df, "auto")

    table = build_residue_report_table(df, score_column=score_column)
    csv_path = out_dir / f"{basename}.csv"
    txt_path = out_dir / f"{basename}.txt"

    table.to_csv(csv_path, index=False)

    lines = [
        "# CryoModel validate — per-residue report (for Coot / manual review)",
        f"# validate_score column: {score_column}",
        "#",
    ]
    for _, row in table.iterrows():
        lines.append(format_residue_report_line(row))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, txt_path
