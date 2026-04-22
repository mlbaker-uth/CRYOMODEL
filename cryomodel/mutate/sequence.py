"""Extract polymer sequence and residues from a gemmi chain."""
from __future__ import annotations

from typing import List, Tuple

import gemmi

def extract_chain_polymer(st: gemmi.Structure, chain_id: str) -> Tuple[str, List[gemmi.Residue]]:
    """Return one-letter sequence and ordered standard residues for a peptide chain."""
    ch = st[0][chain_id]
    poly = ch.get_polymer()
    if poly.length() == 0:
        raise ValueError(f"Chain {chain_id!r} has no polymer.")
    # `make_one_letter_sequence()` can insert '-' for auth/label gaps vs polymer span; use
    # `extract_sequence()` + `one_letter_code` so len(seq) == poly.length() (e.g. 7KJR).
    three = poly.extract_sequence()
    seq = gemmi.one_letter_code(three)
    residues: List[gemmi.Residue] = []
    for i in range(poly.length()):
        residues.append(poly[i])
    if len(seq) != len(residues):
        raise RuntimeError("Sequence length mismatch after polymer extraction.")
    return seq, residues


def assert_homomultimer_same_sequence(st: gemmi.Structure, chain_ids: List[str]) -> str:
    """Require all chains to have identical polymer sequences; return that sequence."""
    seqs = []
    for cid in chain_ids:
        s, _ = extract_chain_polymer(st, cid)
        seqs.append(s)
    if len(set(seqs)) != 1:
        raise ValueError(
            "Homomultimer mode requires identical sequences on all chains; got: "
            + ", ".join(f"{c}={repr(s[:40])}..." if len(s) > 40 else f"{c}={repr(s)}" for c, s in zip(chain_ids, seqs))
        )
    return seqs[0]


def one_letter_to_name(code: str) -> str:
    """Map one-letter code to PDB residue name (standard 20)."""
    c = code.upper()
    names = gemmi.expand_one_letter_sequence(c, gemmi.ResidueKind.AA)
    if len(names) != 1:
        raise ValueError(f"Unknown one-letter code: {code!r}")
    return names[0]
