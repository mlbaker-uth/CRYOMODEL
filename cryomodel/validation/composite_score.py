# cryomodel/validation/composite_score.py
"""Resolution-aware composite quality score and B-factor coloring."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import gemmi
import numpy as np
import pandas as pd

# z-residual columns (from resolution priors) grouped like the composite map/geometry split.
_Z_MAP_RESIDUAL_COLS = (
    "ringer_Z_z_residual",
    "Q_mean_z_residual",
    "CC_mask_z_residual",
    "ZNCC_z_residual",
)
_Z_GEOM_RESIDUAL_COLS = (
    "ramachandran_prob_z_residual",
    "clashscore_z_z_residual",
    "rotamer_prob_z_residual",
    "peptide_twist_score_z_residual",
    "cb_deviation_A_z_residual",
    "rama_penalty_z_residual",
    "molprobity_clashscore_z_residual",
)

def _strictness_from_local_res(
    local_res: pd.Series,
    r_lo: float = 2.0,
    r_hi: float = 7.0,
) -> np.ndarray:
    """Higher in sharper (lower-Å) regions: ~1 at r_lo, ~0 at r_hi and beyond."""
    r = local_res.astype(float).to_numpy()
    s = np.full(r.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(r) & (r > 0)
    s[valid] = np.clip((r_hi - r[valid]) / max(r_hi - r_lo, 1e-6), 0.0, 1.0)
    med = float(np.nanmedian(s[valid])) if np.any(valid) else 0.5
    s[~np.isfinite(s)] = med
    return s


def _pct_bad(series: pd.Series, *, higher_is_better: bool) -> np.ndarray:
    """Map each column to ~[0,1] badness using within-structure percentile ranks."""
    s = series.astype(float)
    ok = s.notna()
    if not ok.any():
        return np.full(len(s), 0.5, dtype=np.float64)
    r = s.rank(pct=True, method="average")
    if higher_is_better:
        bad = 1.0 - r
    else:
        bad = r
    out = bad.to_numpy(dtype=np.float64)
    out[~ok.to_numpy()] = 0.5
    return np.clip(np.nan_to_num(out, nan=0.5), 0.0, 1.0)


def add_composite_quality_columns(
    df: pd.DataFrame,
    r_lo: float = 2.0,
    r_hi: float = 7.0,
) -> pd.DataFrame:
    """Add composite columns including ``composite_quality_0_100`` and ``composite_badness_0_100``.

    Map-derived terms are up-weighted where local resolution is sharper (lower Å);
    blurry regions rely more on geometry. Percentile-based columns are comparable
    within one run. For ChimeraX default B-factor coloring (low = blue, high = red),
    use ``composite_badness_0_100`` or ``composite_band_deviation_0_100`` so red
    means worse. When resolution priors produced ``*_z_residual`` columns,
    ``composite_band_deviation_0_100`` scales mean |z| (resolution-band deviation)
    into 0–100 (~3 mean |z| saturates).
    """
    out = df.copy()
    strict = _strictness_from_local_res(out["local_res"] if "local_res" in out else pd.Series(np.nan, index=out.index), r_lo, r_hi)

    map_terms: list[np.ndarray] = []
    if "Q_mean" in out.columns:
        map_terms.append(_pct_bad(out["Q_mean"], higher_is_better=True))
    if "CC_mask" in out.columns:
        map_terms.append(_pct_bad(out["CC_mask"], higher_is_better=True))
    if "ZNCC" in out.columns:
        map_terms.append(_pct_bad(out["ZNCC"], higher_is_better=True))
    if "ringer_Z" in out.columns:
        map_terms.append(_pct_bad(out["ringer_Z"], higher_is_better=True))

    h1 = out["CC_half1"] if "CC_half1" in out.columns else None
    h2 = out["CC_half2"] if "CC_half2" in out.columns else None
    if h1 is not None and h2 is not None:
        split = (h1.astype(float) - h2.astype(float)).abs()
        has_half = (h1.notna() & h2.notna()) & ((h1.astype(float).abs() + h2.astype(float).abs()) > 1e-6)
        half_bad = _pct_bad(split, higher_is_better=False)
        half_bad[~has_half.to_numpy()] = 0.5
        map_terms.append(half_bad)

    if "CC_delta" in out.columns:
        neg = (-out["CC_delta"].astype(float)).clip(lower=0.0)
        delta_bad = _pct_bad(neg, higher_is_better=False)
        map_terms.append(delta_bad)

    if map_terms:
        map_bad = np.mean(np.stack(map_terms, axis=0), axis=0)
    else:
        map_bad = np.full(len(out), 0.5, dtype=np.float64)

    geom_terms: list[np.ndarray] = []
    if "ramachandran_prob" in out.columns:
        geom_terms.append(_pct_bad(out["ramachandran_prob"], higher_is_better=True))
    if "rotamer_prob" in out.columns:
        geom_terms.append(_pct_bad(out["rotamer_prob"], higher_is_better=True))
    if "clashscore_z" in out.columns:
        geom_terms.append(_pct_bad(out["clashscore_z"], higher_is_better=False))
    if "rama_penalty" in out.columns:
        geom_terms.append(_pct_bad(out["rama_penalty"], higher_is_better=False))

    if geom_terms:
        geom_bad = np.mean(np.stack(geom_terms, axis=0), axis=0)
    else:
        geom_bad = np.full(len(out), 0.5, dtype=np.float64)

    w_map = 0.28 + 0.52 * strict
    w_geom = 1.0 - w_map
    combined = w_map * map_bad + w_geom * geom_bad
    combined = np.clip(combined, 0.0, 1.0)

    out["composite_map_bad"] = map_bad
    out["composite_geom_bad"] = geom_bad
    out["composite_strictness"] = strict
    out["composite_quality_0_100"] = 100.0 * (1.0 - combined)
    # ChimeraX default B-factor ramp: low = blue, high = red → store *badness* so red = worse.
    out["composite_badness_0_100"] = 100.0 - out["composite_quality_0_100"]

    present_map = [c for c in _Z_MAP_RESIDUAL_COLS if c in out.columns]
    present_geom = [c for c in _Z_GEOM_RESIDUAL_COLS if c in out.columns]
    if present_map or present_geom:
        strict_a = out["composite_strictness"].to_numpy(dtype=np.float64)
        w_map = 0.28 + 0.52 * strict_a
        w_geom = 1.0 - w_map

        def _mean_abs_z(sub: pd.DataFrame) -> np.ndarray:
            a = np.abs(sub.to_numpy(dtype=np.float64))
            with np.errstate(invalid="ignore"):
                m = np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=1)
            return np.where(np.isfinite(m), m, 0.0)

        map_m = _mean_abs_z(out[present_map]) if present_map else np.zeros(len(out), dtype=np.float64)
        geom_m = _mean_abs_z(out[present_geom]) if present_geom else np.zeros(len(out), dtype=np.float64)
        if present_map and not present_geom:
            comb = map_m
        elif present_geom and not present_map:
            comb = geom_m
        else:
            comb = w_map * map_m + w_geom * geom_m
        # ~3 mean |z| on the combined scale → saturated “very bad” red.
        z_cap = 3.0
        out["composite_band_deviation_0_100"] = 100.0 * np.clip(comb / z_cap, 0.0, 1.0)
    else:
        out["composite_band_deviation_0_100"] = np.nan

    return out


def resolve_bfactor_column(df: pd.DataFrame, metric: str) -> str:
    """Pick CSV column written to ``atom.b_iso`` for coloring."""
    if metric == "quality":
        return "composite_quality_0_100"
    if metric == "badness":
        return "composite_badness_0_100"
    if metric == "band":
        return "composite_band_deviation_0_100"
    # auto: resolution-band deviation when priors produced z-residuals, else percentile badness
    if "composite_band_deviation_0_100" in df.columns and df["composite_band_deviation_0_100"].notna().any():
        return "composite_band_deviation_0_100"
    return "composite_badness_0_100"


def metric_higher_is_worse(metric: str, column: str) -> bool:
    """Whether larger values mean worse fit (for pass/fail occupancy)."""
    if metric == "quality" or column == "composite_quality_0_100":
        return False
    return True


def apply_composite_bfactors_to_structure(
    structure: gemmi.Structure,
    df: pd.DataFrame,
    column: str,
    *,
    threshold: Optional[float] = None,
    higher_is_worse: bool = True,
    pass_occupancy: float = 1.0,
    fail_occupancy: float = 0.35,
) -> None:
    """Set ``atom.b_iso`` from ``df`` (keyed by ``chain`` + ``seqid``).

    If ``threshold`` is set, also set ``atom.occ``: pass vs fail for a hard visual line
    in ChimeraX/PyMOL (e.g. select ``occupancy < 0.9``). For *higher-is-worse* metrics,
    fail when ``value > threshold``; for *quality*, fail when ``value < threshold``.
    """
    if column not in df.columns:
        raise KeyError(column)
    lookup_b: dict[tuple[str, str], float] = {}
    lookup_occ: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        ch = str(row["chain"])
        sid = str(row["seqid"])
        key = (ch, sid)
        val = row[column]
        if pd.isna(val):
            continue
        v = float(val)
        lookup_b[key] = v
        if threshold is not None:
            if higher_is_worse:
                ok = v <= threshold
            else:
                ok = v >= threshold
            lookup_occ[key] = float(pass_occupancy if ok else fail_occupancy)
        else:
            lookup_occ[key] = float(pass_occupancy)

    for model in structure:
        for chain in model:
            cname = chain.name
            for res in chain:
                key = (cname, str(res.seqid))
                if key not in lookup_b:
                    continue
                b = lookup_b[key]
                occ = lookup_occ.get(key, pass_occupancy)
                for atom in res:
                    atom.b_iso = b
                    atom.occ = occ


def write_structure_with_composite_bfactors(
    structure: gemmi.Structure,
    df: pd.DataFrame,
    out_path: Path,
    *,
    column: str = "composite_badness_0_100",
    format_: Optional[str] = None,
    threshold: Optional[float] = None,
    higher_is_worse: bool = True,
    pass_occupancy: float = 1.0,
    fail_occupancy: float = 0.35,
) -> None:
    """Clone ``structure``, apply B-factors (and optional occupancy pass/fail), write PDB/mmCIF."""
    out_path = Path(out_path)
    st = structure.clone()
    apply_composite_bfactors_to_structure(
        st,
        df,
        column,
        threshold=threshold,
        higher_is_worse=higher_is_worse,
        pass_occupancy=pass_occupancy,
        fail_occupancy=fail_occupancy,
    )
    suf = out_path.suffix.lower()
    fmt = format_
    if fmt is None:
        if suf in (".cif", ".mmcif"):
            fmt = "mmcif"
        else:
            fmt = "pdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mmcif":
        st.write_mmcif(str(out_path))
    else:
        st.write_pdb(str(out_path))
