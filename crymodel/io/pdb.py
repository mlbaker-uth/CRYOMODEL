# crymodel/io/pdb.py
from __future__ import annotations
import numpy as np
import gemmi

def read_model_xyz(path: str, remove_hydrogens: bool = True) -> np.ndarray:
    """Return Nx3 atom coords (x,y,z) in Å from PDB/mmCIF.
    
    Args:
        path: Path to PDB/mmCIF file
        remove_hydrogens: If True, filter out hydrogen atoms (default: True)
    """
    st = gemmi.read_structure(str(path))
    pts = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    # Skip hydrogens if requested
                    if remove_hydrogens:
                        element_name = atom.element.name if atom.element else atom.name.strip()[0] if atom.name.strip() else "C"
                        if element_name.upper() == "H":
                            continue
                    p = atom.pos
                    pts.append((float(p.x), float(p.y), float(p.z)))
    if not pts:
        return np.zeros((0,3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

# --- compatibility alias ---
def read_model(path: str):
    """Compatibility shim: return an object with .xyz np.ndarray (Å)."""
    import numpy as np
    xyz = read_model_xyz(path)  # your existing function
    class _M: pass
    m = _M()
    m.xyz = np.asarray(xyz, dtype=float)
    return m