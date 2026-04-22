"""
Integration tests for symmetry phase-0 on EMD-22898 (C2) + 7KJR model.

Default paths point at the lab examples tree next to the repo; override with:
  CRYOMODEL_SYMMETRY_TEST_MAP
  CRYOMODEL_SYMMETRY_TEST_PDB

Map threshold ~0.5 (absolute) per project convention for this dataset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pytest

from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.preprocess import run_phase0_preprocess

_DEFAULT_MAP = Path("/Users/mbaker-local/Downloads/CRYOMODEL_LOCAL/examples/emd_22898.map")
_DEFAULT_PDB = Path("/Users/mbaker-local/Downloads/CRYOMODEL_LOCAL/examples/7kjr-no-het.pdb")

MAP_PATH = Path(os.environ.get("CRYOMODEL_SYMMETRY_TEST_MAP", str(_DEFAULT_MAP)))
PDB_PATH = Path(os.environ.get("CRYOMODEL_SYMMETRY_TEST_PDB", str(_DEFAULT_PDB)))

pytestmark = pytest.mark.skipif(
    not MAP_PATH.is_file() or not PDB_PATH.is_file(),
    reason=(
        "Gold symmetry fixtures not found; set CRYOMODEL_SYMMETRY_TEST_MAP and "
        f"CRYOMODEL_SYMMETRY_TEST_PDB (looked for {MAP_PATH}, {PDB_PATH})"
    ),
)

# Absolute threshold in raw map units for EMD-22898 in this project.
DENSITY_THRESHOLD = float(os.environ.get("CRYOMODEL_SYMMETRY_TEST_THRESHOLD", "0.5"))


def _chain_ca_com(st: gemmi.Structure, chain_name: str) -> np.ndarray:
    pts: list[list[float]] = []
    for model in st:
        ch = model.find_chain(chain_name)
        if ch is None:
            continue
        for res in ch:
            if res.name == "HOH":
                continue
            try:
                ca = res.sole_atom("CA")
                pts.append([ca.pos.x, ca.pos.y, ca.pos.z])
            except Exception:
                continue
    if not pts:
        raise ValueError(f"No CA atoms for chain {chain_name!r}")
    a = np.array(pts, dtype=np.float64)
    return a.mean(axis=0)


def _dimer_com_direction_normalized(pdb_path: Path, chain_a: str = "A", chain_b: str = "B") -> np.ndarray:
    """Unit vector from chain A CA COM to chain B CA COM (C2-related protomers for 7KJR)."""
    st = gemmi.read_structure(str(pdb_path))
    try:
        st.merge_chain_parts()
    except Exception:
        pass
    ca = _chain_ca_com(st, chain_a)
    cb = _chain_ca_com(st, chain_b)
    d = cb - ca
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        raise ValueError("Degenerate COM separation")
    return d / n


def test_phase0_emd22898_writes_outputs_and_axis(tmp_path: Path):
    out_dir = tmp_path / "sym_out"
    res = run_phase0_preprocess(
        MAP_PATH,
        out_dir=out_dir,
        downsample_factor=4,
        density_threshold=DENSITY_THRESHOLD,
        max_voxels_pca=200_000,
        random_seed=0,
    )
    assert Path(res.output_map).is_file()
    assert Path(res.output_json).is_file()
    with open(res.output_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["density_threshold"] == pytest.approx(DENSITY_THRESHOLD)
    # Count is on the downsampled grid; at 0.5 threshold this map still yields hundreds of voxels.
    assert payload["n_voxels_above_threshold"] > 100
    assert len(payload["principal_axes_xyz"]) == 3
    assert len(payload["principal_axes_xyz"][0]) == 3

    # C2 homodimer (A/B): elongation / first inertia axis of thresholded map should not
    # lie along the inter–COM vector (empirically ~0.007 for this map at thr=0.5).
    d = _dimer_com_direction_normalized(PDB_PATH, "A", "B")
    primary = np.array(res.principal_axes_xyz[0], dtype=np.float64)
    primary /= np.linalg.norm(primary)
    assert abs(float(np.dot(primary, d))) < 0.12

    # Strong elongation along one principal direction (axisymmetric-ish envelope).
    assert payload["eigenvalue_fractions"][0] > 0.45


def test_phase1_emd22898_after_phase0(tmp_path: Path):
    out_dir = tmp_path / "sym_full"
    run_phase0_preprocess(
        MAP_PATH,
        out_dir=out_dir,
        downsample_factor=4,
        density_threshold=DENSITY_THRESHOLD,
        max_voxels_pca=200_000,
        random_seed=0,
    )
    r1 = run_phase1_candidates(
        out_dir,
        tilt_degrees=(0.0, 5.0, 10.0, 15.0),
        include_diagonals=True,
        n_axial_bins=32,
    )
    assert r1.n_candidates >= 15
    assert Path(r1.output_json).is_file()
    pca_cands = [c for c in r1.candidates if str(c["source"]).startswith("pca_axis_0")]
    assert pca_cands, "expected phase-0 primary PCA as a candidate"
    u = np.array(pca_cands[0]["direction_xyz"], dtype=np.float64)
    u /= np.linalg.norm(u)
    # Same convention as phase0 test: elongation ~ Z for EMD-22898
    assert abs(float(u[2])) > 0.85
    for c in r1.candidates:
        assert len(c["axial_mass"]) == 32
        assert c["integrated_mass"] > 0


def test_phase0_emd22898_cli(tmp_path: Path):
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "-m",
        "cryomodel.cli",
        "symmetry",
        "phase0",
        str(MAP_PATH),
        str(out_dir),
        "--downsample",
        "4",
        "--density-threshold",
        str(DENSITY_THRESHOLD),
        "--max-voxels-pca",
        "200000",
        "--seed",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out_dir / "symmetry_phase0.json").is_file()
    assert (out_dir / "symmetry_phase0_downsample.mrc").is_file()


def test_phase1_emd22898_cli(tmp_path: Path):
    out_dir = tmp_path / "cli_p01"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cryomodel.cli",
            "symmetry",
            "phase0",
            str(MAP_PATH),
            str(out_dir),
            "--downsample",
            "4",
            "--density-threshold",
            str(DENSITY_THRESHOLD),
            "--max-voxels-pca",
            "200000",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cryomodel.cli",
            "symmetry",
            "phase1",
            str(out_dir),
            "--tilt-deg",
            "0,5,10",
            "--axial-bins",
            "24",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out_dir / "symmetry_phase1.json").is_file()
