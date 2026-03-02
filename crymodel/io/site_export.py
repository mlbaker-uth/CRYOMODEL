# crymodel/io/site_export.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import csv

class AssignmentSet:
    def __init__(self, centers_xyzA: np.ndarray, meta: dict):
        self.centers_xyzA = np.asarray(centers_xyzA, dtype=np.float32)  # (N,3) in Å
        self.meta = dict(meta)

    def to_json(self) -> str:
        d = {
            "centers_xyzA": self.centers_xyzA.tolist(),
            "meta": self.meta
        }
        return json.dumps(d, indent=2)

def write_json(assigns: AssignmentSet, out_path: Path) -> None:
    out_path.write_text(assigns.to_json())

def write_water_pdb(assigns: AssignmentSet, out_path: Path) -> None:
    """Write waters as HOH with O atoms at assigns.centers_xyzA (Å) to PDB.
    We use a simple, explicit PDB writer to avoid Gemmi version quirks.
    """
    xyz = np.asarray(assigns.centers_xyzA, dtype=np.float32)
    pdb_path = out_path.with_suffix('.pdb')
    with open(pdb_path, 'w') as f:
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1                      \n")
        serial = 1
        chain = 'W'
        resname = 'HOH'
        atom_name = 'O'
        for i, (x, y, z) in enumerate(xyz, start=1):
            line = _pdb_atom_line(serial, atom_name, resname, chain, i, float(x), float(y), float(z), 'O')
            f.write(line + "\n")
            serial += 1
        f.write("TER\nEND\n")

def write_grouped_pseudoatoms_pdb(out_path: Path,
                                  components_xyzA: list[np.ndarray],
                                  chain_prefix: str = "L",
                                  residue_name: str = "LIG",
                                  atom_name: str = "C1") -> None:
    """Write a PDB with one chain per component of pseudoatoms using a simple writer."""
    pdb_path = out_path.with_suffix('.pdb')
    if not components_xyzA:
        pdb_path.write_text("REMARK empty ligand set\n")
        return
    with open(pdb_path, 'w') as f:
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1                      \n")
        serial = 1
        for ci, xyzA in enumerate(components_xyzA):
            # PDB chain IDs are a single character; use A..Z cycling
            chain = chr(ord('A') + (ci % 26))
            for i, (x, y, z) in enumerate(np.asarray(xyzA, dtype=float), start=1):
                line = _pdb_atom_line(serial, atom_name, residue_name, chain, i, float(x), float(y), float(z), 'C')
                f.write(line + "\n")
                serial += 1
            f.write("TER\n")
        f.write("END\n")


def _pdb_atom_line(serial: int,
                   name: str,
                   resname: str,
                   chain: str,
                   resi: int,
                   x: float, y: float, z: float,
                   element: str) -> str:
    """Format a single PDB ATOM record (columns per PDB v3.3)."""
    # name field is left or right justified depending on length; keep simple 4-char padded
    name_field = f"{name:>4s}"[:4]
    return (
        f"ATOM  {serial:5d} {name_field} {resname:>3s} {chain:1s}{resi:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}"
    )


def write_sites_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV with per-site/component features.
    Expects uniform keys across rows; writes header in deterministic key order.
    """
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    # Stable column ordering: id,type,center_x,center_y,center_z,nvox,peak,mean,min_dist_A,... then rest alpha-sorted
    base_cols = [
        "id", "type", "center_x", "center_y", "center_z",
        "nvox", "peak", "mean", "min_dist_A"
    ]
    other_cols = sorted([k for k in rows[0].keys() if k not in base_cols])
    cols = base_cols + other_cols
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
