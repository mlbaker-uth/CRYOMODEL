# crymodel/ml/pdb_reader.py
"""Read PDB/mmCIF files into pandas DataFrames for ML feature extraction."""
from __future__ import annotations
import pandas as pd
import numpy as np
import gemmi


def read_pdb_to_dataframe(path: str, remove_hydrogens: bool = True) -> pd.DataFrame:
    """Read PDB/mmCIF file into pandas DataFrame with atom information.
    
    Args:
        path: Path to PDB/mmCIF file
        remove_hydrogens: If True, filter out hydrogen atoms (default: True)
    
    Returns:
        DataFrame with columns: x, y, z, name, resname, chain, resi, element
    """
    st = gemmi.read_structure(str(path))
    rows = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    # Get element name
                    element_name = atom.element.name if atom.element else atom.name.strip()[0] if atom.name.strip() else "C"
                    
                    # Skip hydrogens if requested
                    if remove_hydrogens and element_name.upper() == "H":
                        continue
                    
                    rows.append({
                        "x": float(atom.pos.x),
                        "y": float(atom.pos.y),
                        "z": float(atom.pos.z),
                        "name": atom.name,
                        "resname": res.name,
                        "chain": chain.name,
                        "resi": res.seqid.num if res.seqid.num is not None else 0,
                        "element": element_name,
                    })
    if not rows:
        return pd.DataFrame(columns=["x", "y", "z", "name", "resname", "chain", "resi", "element"])
    return pd.DataFrame(rows)


def read_candidate_waters_to_dataframe(path: str) -> pd.DataFrame:
    """Read candidate-waters.pdb file into DataFrame with center coordinates.
    
    Returns DataFrame with columns: center_x, center_y, center_z, id (if available)
    """
    st = gemmi.read_structure(str(path))
    rows = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    rows.append({
                        "center_x": float(atom.pos.x),
                        "center_y": float(atom.pos.y),
                        "center_z": float(atom.pos.z),
                    })
    if not rows:
        return pd.DataFrame(columns=["center_x", "center_y", "center_z"])
    df = pd.DataFrame(rows)
    # Add IDs if we can infer them from residue numbers
    if len(df) > 0:
        df["id"] = [f"W{i+1:05d}" for i in range(len(df))]
    return df

