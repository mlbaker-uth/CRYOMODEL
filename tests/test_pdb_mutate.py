"""Tests for pdb-mutate (sequence alignment + side-chain replacement)."""
from __future__ import annotations

from pathlib import Path

import gemmi
from cryomodel.mutate.align import align_global_simple, mutations_from_alignment, mutations_from_aligned_pairs_fasta
from cryomodel.mutate.engine import mutate_pdb

FIX = Path(__file__).resolve().parent / "fixtures"


def test_align_global_simple_no_gaps():
    a, b = align_global_simple("ACDEF", "ACDEF")
    assert a == b == "ACDEF"


def test_mutations_from_alignment_tiny():
    pdb_seq = "AGS"
    aln_p, aln_t = align_global_simple(pdb_seq, "AGA")
    residues = ["r0", "r1", "r2"]
    pairs = mutations_from_alignment(pdb_seq, residues, aln_p, aln_t)
    assert len(pairs) == 1
    assert pairs[0] == ("r2", "A")


def test_mutations_from_alignment_skips_target_gap_columns():
    """Target gap columns keep the PDB residue; no substitution."""
    pdb_seq = "ACE"
    residues = ["r0", "r1", "r2"]
    pairs = mutations_from_alignment(pdb_seq, residues, "ACE", "A-E")
    assert pairs == []


def test_mutations_from_aligned_pairs_fasta_target_gaps():
    pdb_seq = "MKTV"
    residues = [f"r{i}" for i in range(4)]
    pairs = mutations_from_aligned_pairs_fasta(pdb_seq, residues, FIX / "mutate_align_target_gaps.fasta")
    assert pairs == []


def test_mutations_from_aligned_pairs_fasta_swapped_record_order():
    """Target row first, PDB row second — same as mutate_align_target_gaps.fasta but swapped."""
    pdb_seq = "MKTV"
    residues = [f"r{i}" for i in range(4)]
    pairs = mutations_from_aligned_pairs_fasta(pdb_seq, residues, FIX / "mutate_align_swapped_order.fasta")
    assert pairs == []


def test_mutations_from_aligned_pairs_fasta_extra_template_residues():
    """Template row contains residues not in the coordinate chain (skipped until match)."""
    pdb_seq = "MKTV"
    residues = [f"r{i}" for i in range(4)]
    pairs = mutations_from_aligned_pairs_fasta(pdb_seq, residues, FIX / "mutate_align_extra_template_prefix.fasta")
    assert pairs == []


def test_mutations_from_aligned_pairs_fasta_msa_row_indices():
    """Pick rows 1 and 2 from a multi-sequence FASTA (row 0 ignored)."""
    pdb_seq = "MKTV"
    residues = [f"r{i}" for i in range(4)]
    pairs = mutations_from_aligned_pairs_fasta(
        pdb_seq, residues, FIX / "mutate_align_three_records.fasta", alignment_row_a=1, alignment_row_b=2
    )
    assert pairs == []


def test_mutations_from_aligned_pairs_fasta_target_gaps_partial_mutation():
    pdb_seq = "ACE"
    residues = ["r0", "r1", "r2"]
    pairs = mutations_from_aligned_pairs_fasta(
        pdb_seq, residues, FIX / "mutate_align_target_gaps_mutation.fasta"
    )
    assert len(pairs) == 1
    assert pairs[0] == ("r1", "M")


def test_mutate_pdb_writes_expected_mutation():
    out = FIX / "_tmp_mut_out.pdb"
    try:
        r = mutate_pdb(
            str(FIX / "tiny_peptide.pdb"),
            str(out),
            ["A"],
            target_fasta=str(FIX / "mutate_target_aga.fasta"),
        )
        assert len(r.mutations) == 1
        assert r.mutations[0]["to"] == "A"
        g = r.mutations[0]["guide"]
        assert "before" in g and "after" in g and "delta" in g
        assert "clash" in g["before"] and "delta_clash" in g["delta"]
        assert r.map_guide_reference is None
        st = gemmi.read_structure(str(out))
        res3 = next(r for r in st[0]["A"] if str(r.seqid.num) == "3")
        assert res3.name == "ALA"
    finally:
        if out.exists():
            out.unlink()
