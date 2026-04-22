"""Small Gemmi helpers shared across cryomodel (atom lookup by name)."""
from __future__ import annotations

from typing import Optional

import gemmi


def sole_atom(residue: gemmi.Residue, name: str) -> Optional[gemmi.Atom]:
    """Return the uniquely named atom in ``residue``, or None if missing / ambiguous."""
    try:
        return residue.sole_atom(name)
    except Exception:
        return None
