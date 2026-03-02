# crymodel/io/debug_export.py
from pathlib import Path
import numpy as np
from ..core.types import ModelAtoms
from .pdb import write_model
from .pdb_simple import write_pdb_atoms_simple

def _atoms_from_xyz(xyz_ang: np.ndarray, resname="UNX", atom_name="X", element="C") -> ModelAtoms:
    arr = np.asarray(xyz_ang, dtype=np.float32).reshape(-1, 3)
    n = int(len(arr))
    return ModelAtoms(
        xyz=arr,
        name=np.array([atom_name]*n, dtype=object),
        resname=np.array([resname]*n, dtype=object),
        chain=np.array(["X"]*n, dtype=object),
        resi=np.arange(1, n+1, dtype=int),
        element=np.array(["C"]*n, dtype=object),
    )

def export_centers_all(xyz_ang, out_base: Path, resname="UNX"):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(xyz_ang, dtype=np.float32).reshape(-1, 3)
    # PDB
    write_pdb_atoms_simple(str(out_base.with_suffix(".pdb")), arr, resname=resname, atom_name="X", chain_id="X")
    # CSV
    import csv
    with open(out_base.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_A", "y_A", "z_A"])
        for x, y, z in arr:
            w.writerow([f"{float(x):.3f}", f"{float(y):.3f}", f"{float(z):.3f}"])

def export_seeds_all(seeds_ang, out_base: Path):
    export_centers_all(seeds_ang, out_base, resname="SEED")