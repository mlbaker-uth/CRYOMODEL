# cryomodel/io/structure_filter.py
"""Structure filtering helpers (e.g. protein-only validation)."""
from __future__ import annotations

import gemmi


def is_tabulated_amino_acid_residue(residue: gemmi.Residue) -> bool:
    """True if Gemmi recognizes the residue as a standard/tabulated amino acid."""
    info = gemmi.find_tabulated_residue(residue.name)
    return bool(info.found() and info.is_amino_acid())


def filter_protein_only(structure: gemmi.Structure) -> gemmi.Structure:
    """Return a deep copy with only tabulated amino-acid polymer residues kept.

    Waters, ions, nucleic acids, and ligands are removed. Empty chains are dropped.
    """
    st = structure.clone()
    for model in st:
        chains_to_delete: list[int] = []
        for ci in range(len(model)):
            chain = model[ci]
            for ri in range(len(chain) - 1, -1, -1):
                res = chain[ri]
                if not is_tabulated_amino_acid_residue(res):
                    del chain[ri]
            if len(chain) == 0:
                chains_to_delete.append(ci)
        for ci in reversed(chains_to_delete):
            del model[ci]
    return st
