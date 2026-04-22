from __future__ import annotations

import csv
import json
from pathlib import Path

import gemmi

from cryomodel.conservation import compute_conservation
from cryomodel.mutate.sequence import extract_chain_polymer


FIX = Path(__file__).resolve().parent / "fixtures"


def test_compute_conservation_outputs_and_metrics(tmp_path: Path):
    out_csv = tmp_path / "conservation.csv"
    out_json = tmp_path / "conservation.json"
    out_pdb = tmp_path / "conservation_b.pdb"

    result = compute_conservation(
        pdb_path=FIX / "tiny_peptide.pdb",
        chains="A",
        alignment_fasta=FIX / "conservation_tiny.fasta",
        out_csv=out_csv,
        out_json=out_json,
        out_pdb=out_pdb,
        bfactor_metric="n_aa_types",
        occupancy_metric="p_nonref",
        include_reference_in_stats=False,
    )

    assert len(result.rows) == 3
    by_seqid = {int(r["seqid"]): r for r in result.rows}
    # seqid 1: A/V/A -> 2 unique, 1/3 non-ref
    assert by_seqid[1]["n_aa_types"] == 2
    assert abs(float(by_seqid[1]["p_nonref"]) - (1.0 / 3.0)) < 1e-6
    # seqid 2: G/G/- -> 1 unique, gap fraction 1/3
    assert by_seqid[2]["n_aa_types"] == 1
    assert abs(float(by_seqid[2]["p_gap"]) - (1.0 / 3.0)) < 1e-6
    # seqid 3: A/S/S (ref is S) -> 2 unique, 1/3 non-ref
    assert by_seqid[3]["n_aa_types"] == 2
    assert abs(float(by_seqid[3]["p_nonref"]) - (1.0 / 3.0)) < 1e-6

    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["chains"] == ["A"]
    assert len(payload["rows"]) == 3

    st = gemmi.read_structure(str(out_pdb))
    r1 = next(r for r in st[0]["A"] if int(r.seqid.num) == 1)
    assert abs(r1.get_ca().b_iso - 2.0) < 1e-6
    # PDB text output rounds occupancy (typically to 2 decimals).
    assert abs(r1.get_ca().occ - (1.0 / 3.0)) < 5e-3


def test_compute_conservation_homomultimer_two_chains(tmp_path: Path):
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    ch_b = gemmi.Chain("B")
    for res in st[0]["A"]:
        ch_b.add_residue(res.clone())
    st[0].add_chain(ch_b)
    dimer_pdb = tmp_path / "dimer.pdb"
    st.write_pdb(str(dimer_pdb))

    out_csv = tmp_path / "c.csv"
    result = compute_conservation(
        pdb_path=dimer_pdb,
        chains="A,B",
        alignment_fasta=FIX / "conservation_tiny.fasta",
        out_csv=out_csv,
        out_json=None,
        out_pdb=None,
        include_reference_in_stats=False,
    )
    assert len(result.rows) == 6
    by_chain = {r["chain"]: [] for r in result.rows}
    for r in result.rows:
        by_chain[r["chain"]].append(int(r["seqid"]))
    assert sorted(by_chain["A"]) == [1, 2, 3]
    assert sorted(by_chain["B"]) == [1, 2, 3]
    _, ra = extract_chain_polymer(gemmi.read_structure(str(dimer_pdb)), "A")
    _, rb = extract_chain_polymer(gemmi.read_structure(str(dimer_pdb)), "B")
    assert ra[0].seqid == rb[0].seqid


def test_compute_conservation_reference_subsequence_prefix(tmp_path: Path):
    """Reference row has extra leading residues vs the PDB (same metrics as tiny alignment)."""
    out_csv = tmp_path / "c.csv"
    result = compute_conservation(
        pdb_path=FIX / "tiny_peptide.pdb",
        chains="A",
        alignment_fasta=FIX / "conservation_subseq_ref.fasta",
        out_csv=out_csv,
        out_json=None,
        out_pdb=None,
        include_reference_in_stats=False,
    )
    assert len(result.rows) == 3
    by_seqid = {int(r["seqid"]): r for r in result.rows}
    assert by_seqid[1]["n_aa_types"] == 2
    assert abs(float(by_seqid[1]["p_nonref"]) - (1.0 / 3.0)) < 1e-6

