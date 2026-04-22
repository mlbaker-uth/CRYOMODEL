# cryomodel/validation/benchmark_priors.py
"""Resolution-binned geometry expectations from ModBench-style benchmark JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_MODALITIES = frozenset({"xray", "em", "combined", "all"})


def default_benchmark_json_path() -> Path:
    """Bundled ``benchmark_data.json`` (0.1 Å bins, xray/em/combined/all)."""
    return Path(__file__).resolve().parent.parent / "data" / "benchmark_data.json"


def load_benchmark_table(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or default_benchmark_json_path()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _outlier_pct_to_rama_median(mean_pct: float) -> float:
    """Map deposited mean Ramachandran outlier % to a target ``ramachandran_prob`` median (~0–1, higher better)."""
    o = max(0.0, float(mean_pct) / 100.0)
    return float(max(0.12, min(0.995, 1.0 - 1.15 * o)))


def _outlier_pct_to_rotamer_median(mean_pct: float) -> float:
    o = max(0.0, float(mean_pct) / 100.0)
    return float(max(0.12, min(0.995, 1.0 - 1.0 * o)))


def _sd_pct_to_mad(sd_pct: float, *, floor: float = 0.05) -> float:
    """Turn benchmark SD (in percent points) into a robust MAD scale for z-scores."""
    s = max(0.0, float(sd_pct) / 100.0)
    return float(max(floor, s / 1.4826 if s > 1e-9 else floor))


def priors_dict_from_benchmark(
    modality: str = "em",
    *,
    json_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build a priors dict compatible with :func:`resolution_priors.compute_z_residuals`.

    Uses 0.1 Å bins from the benchmark. For each bin, sets ``ramachandran_prob``,
    ``rotamer_prob``, and ``molprobity_clashscore`` (from ``clashscore_median`` /
    ``clashscore_sd``) for comparison to :func:`geometry_priors.molprobity_like_clashscore_heavy`.
    Other feature columns are omitted so YAML / :func:`fit_resolution_priors` can still
    supply map-derived metrics for the same bins when merged.

    Args:
        modality: ``em``, ``xray``, ``combined``, or ``all`` (JSON top-level keys).
        json_path: Optional path to a benchmark JSON (defaults to bundled file).
    """
    m = modality.strip().lower()
    if m not in _MODALITIES:
        raise ValueError(f"modality must be one of {sorted(_MODALITIES)}, got {modality!r}")

    data = load_benchmark_table(json_path)
    if m not in data:
        raise KeyError(f"No key {m!r} in benchmark JSON (have {list(data.keys())})")

    rows = data[m]
    priors: Dict[str, Dict[str, Dict[str, float]]] = {}

    for row in rows:
        key = str(row["resolution_bin"])
        ro_m = float(row.get("ramachandran_outliers_mean_percent", 0.0))
        ro_sd = float(row.get("ramachandran_outliers_sd_percent", 0.0))
        rt_m = float(row.get("rotamer_outliers_mean_percent", 0.0))
        rt_sd = float(row.get("rotamer_outliers_sd_percent", 0.0))

        rama_med = _outlier_pct_to_rama_median(ro_m)
        rama_mad = _sd_pct_to_mad(ro_sd, floor=0.06)
        rot_med = _outlier_pct_to_rotamer_median(rt_m)
        rot_mad = _sd_pct_to_mad(rt_sd, floor=0.06)

        cs_med = float(row.get("clashscore_median", row.get("clashscore_mean", 0.0)))
        cs_sd = float(row.get("clashscore_sd", 0.0))
        cs_mad = float(max(cs_sd / 1.4826 if cs_sd > 1e-9 else 0.0, 0.15))

        priors[key] = {
            "ramachandran_prob": {
                "median": rama_med,
                "mad": rama_mad,
                "mean": rama_med,
                "std": rama_mad * 1.4826,
            },
            "rotamer_prob": {
                "median": rot_med,
                "mad": rot_mad,
                "mean": rot_med,
                "std": rot_mad * 1.4826,
            },
            "molprobity_clashscore": {
                "median": cs_med,
                "mad": cs_mad,
                "mean": float(row.get("clashscore_mean", cs_med)),
                "std": cs_sd,
            },
        }

    return priors
