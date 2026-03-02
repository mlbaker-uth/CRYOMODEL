# crymodel/ml/extract_features.py
"""Extract features from PDB files for training."""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import gemmi

from .pdb_reader import read_pdb_to_dataframe
from .featureizer import build_feature_row, build_kdtree


# Map PDB chemical IDs to model classes
ION_CLASS_MAP = {
    "HOH": "HOH",
    "WAT": "HOH",  # alternative water name
    "CL": "Cl",
    "ZN": "Zn",
    "CA": "Ca",
    "MG": "Mg",
    "NA": "Na",
    "K": "K",
    "MN": "Mn",
    "FE": "Fe2",  # default to Fe2+; could be refined with oxidation state
    "FE2": "Fe2",
    "FE3": "Fe3",
    "NI": "Ni",  # not in model classes, but could add
    "CD": "Cd",  # not in model classes
    "CU": "Cu",  # not in model classes
    "BR": "Br",  # not in model classes
    "CO": "Co",  # not in model classes
}

# Model classes we support
SUPPORTED_CLASSES = {"HOH", "Na", "K", "Mg", "Ca", "Mn", "Fe2", "Fe3", "Cl", "Zn"}


def get_ion_label_from_pdb(pdb_path: Path, remove_hydrogens: bool = True) -> pd.DataFrame:
    """Extract ion/water coordinates and labels from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        remove_hydrogens: If True, filter out hydrogens
        
    Returns:
        DataFrame with columns: x, y, z, label, pdb_id, resname, chain, resi
    """
    st = gemmi.read_structure(str(pdb_path))
    pdb_id = pdb_path.stem.upper()
    
    rows = []
    for model in st:
        for chain in model:
            for res in chain:
                # Check if this is a water/ion residue
                resname = res.name.strip().upper()
                if resname in ION_CLASS_MAP:
                    label = ION_CLASS_MAP[resname]
                    if label not in SUPPORTED_CLASSES:
                        continue  # Skip unsupported ions
                    
                    for atom in res:
                        # Skip hydrogens if requested
                        if remove_hydrogens:
                            element_name = atom.element.name if atom.element else atom.name.strip()[0] if atom.name.strip() else "C"
                            if element_name.upper() == "H":
                                continue
                        
                        rows.append({
                            "x": float(atom.pos.x),
                            "y": float(atom.pos.y),
                            "z": float(atom.pos.z),
                            "label": label,
                            "pdb_id": pdb_id,
                            "resname": resname,
                            "chain": chain.name,
                            "resi": res.seqid.num if res.seqid.num is not None else 0,
                        })
    
    if not rows:
        return pd.DataFrame(columns=["x", "y", "z", "label", "pdb_id", "resname", "chain", "resi"])
    return pd.DataFrame(rows)


def extract_features_for_pdb(
    pdb_path: Path,
    resolution: Optional[float] = None,
    remove_hydrogens: bool = True,
) -> pd.DataFrame:
    """Extract features for all ions/waters in a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        resolution: Optional resolution value
        remove_hydrogens: If True, filter out hydrogens
        
    Returns:
        DataFrame with features and labels
    """
    # Get ion/water labels
    ion_df = get_ion_label_from_pdb(pdb_path, remove_hydrogens=remove_hydrogens)
    
    if len(ion_df) == 0:
        return pd.DataFrame()
    
    # Load full model for feature extraction
    atom_df = read_pdb_to_dataframe(str(pdb_path), remove_hydrogens=remove_hydrogens)
    if len(atom_df) == 0:
        return pd.DataFrame()
    
    kdtree, coords = build_kdtree(atom_df)
    
    # Extract features for each ion/water
    features = []
    for _, row in ion_df.iterrows():
        # Create a row dict compatible with build_feature_row
        feat_row = {
            "center_x": row["x"],
            "center_y": row["y"],
            "center_z": row["z"],
        }
        
        feat_dict = build_feature_row(
            feat_row,
            atom_df,
            kdtree,
            coords,
            extra_cols=None,
            entry_resolution=resolution,
        )
        
        # Add label and metadata
        feat_dict["label"] = row["label"]
        feat_dict["pdb_id"] = row["pdb_id"]
        feat_dict["resname"] = row["resname"]
        feat_dict["chain"] = row["chain"]
        feat_dict["resi"] = row["resi"]
        feat_dict["x"] = row["x"]
        feat_dict["y"] = row["y"]
        feat_dict["z"] = row["z"]
        
        features.append(feat_dict)
    
    return pd.DataFrame(features)


def extract_features_batch(
    pdb_dir: Path,
    output_csv: Path,
    pdb_ids: Optional[list[str]] = None,
    resolution_csv: Optional[Path] = None,
    remove_hydrogens: bool = True,
    max_structures: Optional[int] = None,
) -> pd.DataFrame:
    """Extract features for multiple PDB files.
    
    Args:
        pdb_dir: Directory containing PDB files
        output_csv: Path to write output CSV
        pdb_ids: Optional list of PDB IDs to process (if None, process all)
        resolution_csv: Optional CSV with pdb_id and resolution columns
        remove_hydrogens: If True, filter out hydrogens
        max_structures: Optional limit on number of structures to process
        
    Returns:
        Combined DataFrame with all features
    """
    # Load resolution data if provided
    resolution_dict = {}
    if resolution_csv and resolution_csv.exists():
        res_df = pd.read_csv(resolution_csv)
        if "pdb_id" in res_df.columns and "resolution" in res_df.columns:
            resolution_dict = dict(zip(res_df["pdb_id"].str.upper(), res_df["resolution"]))
    
    # Get list of PDB files to process
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if pdb_ids:
        pdb_ids_upper = {pid.upper() for pid in pdb_ids}
        pdb_files = [f for f in pdb_files if f.stem.upper() in pdb_ids_upper]
    
    if max_structures:
        pdb_files = pdb_files[:max_structures]
    
    print(f"Processing {len(pdb_files)} PDB files...")
    import sys
    sys.stdout.flush()
    
    all_features = []
    failed = []
    
    for i, pdb_file in enumerate(pdb_files, 1):
        try:
            pdb_id = pdb_file.stem.upper()
            resolution = resolution_dict.get(pdb_id)
            
            feat_df = extract_features_for_pdb(
                pdb_file,
                resolution=resolution,
                remove_hydrogens=remove_hydrogens,
            )
            
            if len(feat_df) > 0:
                all_features.append(feat_df)
                print(f"  [{i}/{len(pdb_files)}] {pdb_id}: {len(feat_df)} ions/waters", flush=True)
            else:
                print(f"  [{i}/{len(pdb_files)}] {pdb_id}: No ions/waters found", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(pdb_files)}] {pdb_file.stem}: ERROR - {e}", flush=True)
            failed.append(pdb_file.stem)
        
        # Save incrementally every 100 files
        if i % 100 == 0 and all_features:
            temp_df = pd.concat(all_features, ignore_index=True)
            temp_df.to_csv(output_csv, index=False)
            print(f"  [Progress] Saved {len(temp_df)} samples from {len(all_features)} structures so far...", flush=True)
    
    if failed:
        print(f"\nFailed to process {len(failed)} structures: {failed}")
    
    if not all_features:
        print("No features extracted!")
        return pd.DataFrame()
    
    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"\nExtracted features for {len(combined_df)} ions/waters from {len(all_features)} structures")
    print(f"Saved to {output_csv}")
    
    # Print class distribution
    print("\nClass distribution:")
    print(combined_df["label"].value_counts())
    
    return combined_df

