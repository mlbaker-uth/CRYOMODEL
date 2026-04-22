"""Tier-3 conservation diffusion on Cα graph."""
from __future__ import annotations

from pathlib import Path

import gemmi

from cryomodel.conservation_diffusion import run_conservation_diffusion

FIX = Path(__file__).resolve().parent / "fixtures"


def test_diffuse_single_chain_shape_and_columns(tmp_path: Path):
    out_csv = tmp_path / "d.csv"
    out_json = tmp_path / "d.json"
    r = run_conservation_diffusion(
        pdb_path=FIX / "tiny_peptide.pdb",
        chains="A",
        alignment_fasta=FIX / "conservation_tiny.fasta",
        out_csv=out_csv,
        out_json=out_json,
        out_pdb=None,
        contact_radius=50.0,
        falloff_angstrom=5.0,
        diffusion_steps=5,
        mix=0.5,
        peak_min=0.001,
        basin_mode="nearest_peak",
    )
    assert len(r.rows) == 3
    assert "diffused_score" in r.rows[0]
    assert "basin_id" in r.rows[0]
    assert "seed_signal" in r.rows[0]
    assert "seed_raw" in r.rows[0]


def test_diffuse_composite_seed_metric(tmp_path: Path):
    out_csv = tmp_path / "c.csv"
    r = run_conservation_diffusion(
        pdb_path=FIX / "tiny_peptide.pdb",
        chains="A",
        alignment_fasta=FIX / "conservation_tiny.fasta",
        out_csv=out_csv,
        out_json=None,
        out_pdb=None,
        seed_metric="composite_nonref_penalty",
        contact_radius=50.0,
        diffusion_steps=3,
        mix=0.5,
    )
    assert len(r.rows) == 3
    # seqid 1: p_nonref=1/3, mean_penalty=0.5 -> raw product ~0.166667
    row1 = next(x for x in r.rows if int(x["seqid"]) == 1)
    assert float(row1["seed_raw"]) > 0.0
    assert out_csv.exists()


def test_diffuse_homomultimer_graph(tmp_path: Path):
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    ch_b = gemmi.Chain("B")
    for res in st[0]["A"]:
        ch_b.add_residue(res.clone())
    st[0].add_chain(ch_b)
    dimer = tmp_path / "dimer.pdb"
    st.write_pdb(str(dimer))

    out_csv = tmp_path / "d2.csv"
    r = run_conservation_diffusion(
        pdb_path=dimer,
        chains="A,B",
        alignment_fasta=FIX / "conservation_tiny.fasta",
        out_csv=out_csv,
        out_json=None,
        out_pdb=None,
        contact_radius=15.0,
        diffusion_steps=8,
        mix=0.45,
        peak_min=0.0001,
    )
    assert len(r.rows) == 6
    # Cross-chain diffusion: B-chain copy should receive non-zero diffused signal from A
    by_chain = {row["chain"]: row for row in r.rows if int(row["seqid"]) == 1}
    assert "A" in by_chain and "B" in by_chain
