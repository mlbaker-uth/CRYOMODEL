"""ModBench-style benchmark JSON → resolution priors."""
from __future__ import annotations

import pandas as pd

from cryomodel.validation.benchmark_priors import default_benchmark_json_path, priors_dict_from_benchmark
from cryomodel.validation.resolution_priors import compute_z_residuals, merge_resolution_priors


def test_bundled_benchmark_json_exists() -> None:
    p = default_benchmark_json_path()
    assert p.is_file(), f"missing {p}"


def test_priors_dict_from_benchmark_em() -> None:
    priors = priors_dict_from_benchmark("em")
    assert len(priors) >= 30
    assert "3.0-3.1" in priors
    assert "ramachandran_prob" in priors["3.0-3.1"]
    assert "rotamer_prob" in priors["3.0-3.1"]
    assert "molprobity_clashscore" in priors["3.0-3.1"]
    mp = priors["3.0-3.1"]["molprobity_clashscore"]
    assert mp["median"] >= 0 and mp["mad"] > 0
    rp = priors["3.0-3.1"]["ramachandran_prob"]
    assert "median" in rp and "mad" in rp
    assert 0.0 < rp["median"] <= 1.0
    assert rp["mad"] > 0


def test_compute_z_residuals_prefers_narrowest_bin() -> None:
    priors = {
        "2.0-2.2": {"Q_mean": {"median": 0.2, "mad": 0.1}},
        "2.0-2.1": {"Q_mean": {"median": 0.8, "mad": 0.1}},
    }
    df = pd.DataFrame([{"local_res": 2.05, "Q_mean": 0.8}])
    out = compute_z_residuals(df, priors)
    assert abs(float(out["Q_mean_z_residual"].iloc[0])) < 1e-6


def test_merge_resolution_priors_overlay_wins() -> None:
    base = {"2.0-2.1": {"Q_mean": {"median": 0.1, "mad": 0.05}}}
    over = {"2.0-2.1": {"ramachandran_prob": {"median": 0.9, "mad": 0.1}}}
    m = merge_resolution_priors(base, over)
    assert m["2.0-2.1"]["Q_mean"]["median"] == 0.1
    assert m["2.0-2.1"]["ramachandran_prob"]["median"] == 0.9
