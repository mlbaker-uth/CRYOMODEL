from pathlib import Path
import numpy as np

def write_pdb_atoms_simple(path: str, xyz_ang, resname="UNX", atom_name="X", chain_id="X"):
    arr = np.asarray(xyz_ang, dtype=np.float32).reshape(-1, 3)
    n = int(len(arr))
    lines = []
    for i, (x, y, z) in enumerate(arr, start=1):
        lines.append(
            f"ATOM  {i:5d} {atom_name:<4s}{resname:>3s} {chain_id}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
        )
    if n > 0:
        lines.append("TER\n")
    lines.append("END\n")
    Path(path).write_text("".join(lines))
    return n
