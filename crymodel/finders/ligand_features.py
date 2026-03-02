# crymodel/finders/ligand_features.py
"""Feature extraction for ligand components."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from scipy.linalg import eigh
from typing import Optional
import pandas as pd

from ..io.mrc import MapVolume
from ..io.pdb import read_model_xyz


def read_ligand_components_pdb(pdb_path: str) -> list[tuple[str, np.ndarray]]:
    """Read ligand components from PDB file.
    
    Each chain in the PDB represents one component.
    Returns list of (chain_id, xyz_array) tuples.
    """
    components = []
    current_chain = None
    current_coords = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21:22].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if current_chain is None:
                    current_chain = chain
                elif current_chain != chain:
                    # New chain - save previous component
                    if current_coords:
                        components.append((current_chain, np.array(current_coords, dtype=np.float32)))
                    current_chain = chain
                    current_coords = []
                
                current_coords.append([x, y, z])
            elif line.startswith("TER"):
                # End of chain - save component
                if current_chain is not None and current_coords:
                    components.append((current_chain, np.array(current_coords, dtype=np.float32)))
                    current_chain = None
                    current_coords = []
    
    # Save last component
    if current_chain is not None and current_coords:
        components.append((current_chain, np.array(current_coords, dtype=np.float32)))
    
    return components


def extract_component_mask(
    component_xyz: np.ndarray,
    ligand_map: MapVolume,
    padding_A: float = 6.0,
    threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Extract binary mask for a component from the ligand map.
    
    Uses connected components in the ligand map near the pseudoatoms.
    
    Args:
        component_xyz: (N, 3) array of pseudoatom coordinates in Å
        ligand_map: MapVolume with ligand density
        padding_A: Padding around bbox in Å
        threshold: Density threshold for connected components
        
    Returns:
        mask_zyx: Binary mask (z, y, x) for this component
        voxel_indices: (M, 3) array of voxel indices (z, y, x)
    """
    apix = ligand_map.apix
    origin = ligand_map.origin_xyzA
    
    # Convert Å to voxel coordinates
    # Map: (x, y, z) Å -> (z, y, x) voxels
    xyz_vox = (component_xyz - origin) / apix
    zyx_vox = xyz_vox[:, [2, 1, 0]]  # Convert to (z, y, x)
    
    # Get bounding box with padding
    z_min, y_min, x_min = zyx_vox.min(axis=0) - padding_A / apix
    z_max, y_max, x_max = zyx_vox.max(axis=0) + padding_A / apix
    
    z_min, y_min, x_min = int(np.floor(z_min)), int(np.floor(y_min)), int(np.floor(x_min))
    z_max, y_max, x_max = int(np.ceil(z_max)), int(np.ceil(y_max)), int(np.ceil(x_max))
    
    # Clip to map bounds
    z_min = max(0, z_min)
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    z_max = min(ligand_map.data_zyx.shape[0], z_max)
    y_max = min(ligand_map.data_zyx.shape[1], y_max)
    x_max = min(ligand_map.data_zyx.shape[2], x_max)
    
    if z_max <= z_min or y_max <= y_min or x_max <= x_min:
        # Empty bbox
        full_mask = np.zeros(ligand_map.data_zyx.shape, dtype=bool)
        return full_mask, np.zeros((0, 3), dtype=int)
    
    # Extract subvolume
    subvol = ligand_map.data_zyx[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Find connected components in subvolume
    from scipy.ndimage import label
    mask_subvol = subvol > threshold
    if mask_subvol.sum() == 0:
        # No density in this region
        full_mask = np.zeros(ligand_map.data_zyx.shape, dtype=bool)
        return full_mask, np.zeros((0, 3), dtype=int)
    
    labels, n_labels = label(mask_subvol, structure=np.ones((3, 3, 3)))
    
    # Find which label(s) contain the pseudoatoms
    component_labels = set()
    radius_vox = 3.0 / apix  # 3Å radius to find nearby voxels
    for z, y, x in zyx_vox:
        iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
        iz_local = iz - z_min
        iy_local = iy - y_min
        ix_local = ix - x_min
        
        if (0 <= iz_local < labels.shape[0] and 
            0 <= iy_local < labels.shape[1] and 
            0 <= ix_local < labels.shape[2]):
            # Check nearby voxels
            z_start = max(0, iz_local - int(radius_vox))
            z_end = min(labels.shape[0], iz_local + int(radius_vox) + 1)
            y_start = max(0, iy_local - int(radius_vox))
            y_end = min(labels.shape[1], iy_local + int(radius_vox) + 1)
            x_start = max(0, ix_local - int(radius_vox))
            x_end = min(labels.shape[2], ix_local + int(radius_vox) + 1)
            
            nearby_labels = labels[z_start:z_end, y_start:y_end, x_start:x_end]
            unique_labels = np.unique(nearby_labels)
            component_labels.update(unique_labels[unique_labels > 0])  # Exclude background (0)
    
    # Create mask from selected labels
    if not component_labels:
        # No connected component found - use simple sphere around pseudoatoms
        mask_subvol = np.zeros(subvol.shape, dtype=bool)
        for z, y, x in zyx_vox:
            iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
            iz_local = iz - z_min
            iy_local = iy - y_min
            ix_local = ix - x_min
            if (0 <= iz_local < mask_subvol.shape[0] and 
                0 <= iy_local < mask_subvol.shape[1] and 
                0 <= ix_local < mask_subvol.shape[2]):
                # Mark small region around point
                z_start = max(0, iz_local - 2)
                z_end = min(mask_subvol.shape[0], iz_local + 3)
                y_start = max(0, iy_local - 2)
                y_end = min(mask_subvol.shape[1], iy_local + 3)
                x_start = max(0, ix_local - 2)
                x_end = min(mask_subvol.shape[2], ix_local + 3)
                mask_subvol[z_start:z_end, y_start:y_end, x_start:x_end] = True
    else:
        # Use connected component mask
        mask_subvol = np.isin(labels, list(component_labels))
    
    # Create full-size mask
    full_mask = np.zeros(ligand_map.data_zyx.shape, dtype=bool)
    full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = mask_subvol
    
    # Get voxel indices
    voxel_indices = np.array(np.where(full_mask)).T
    
    return full_mask, voxel_indices


def compute_size_features(
    component_xyz: np.ndarray,
    component_mask: np.ndarray,
    ligand_map: MapVolume
) -> dict[str, float]:
    """Compute size and compactness features.
    
    Args:
        component_xyz: (N, 3) pseudoatom coordinates in Å
        component_mask: Binary mask (z, y, x) for this component
        ligand_map: MapVolume with ligand density
        
    Returns:
        Dictionary of size features
    """
    apix = ligand_map.apix
    
    # Volume in Å³
    n_voxels = component_mask.sum()
    volume_A3 = n_voxels * (apix ** 3)
    
    # Number of pseudoatoms
    n_points = len(component_xyz)
    
    # Center of mass
    if n_points > 0:
        com = component_xyz.mean(axis=0)
    else:
        com = np.array([0.0, 0.0, 0.0])
    
    # Radius of gyration
    if n_points > 1:
        centered = component_xyz - com
        Rg = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    else:
        Rg = 0.0
    
    # PCA on pseudoatoms
    if n_points >= 3:
        centered = component_xyz - com
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = eigh(cov)
        eigenvals = np.sort(eigenvals)[::-1]  # λ1 ≥ λ2 ≥ λ3
        anisotropy = (eigenvals[0] - eigenvals[2]) / (eigenvals[0] + 1e-6) if eigenvals[0] > 1e-6 else 0.0
        planarity = 1.0 - (eigenvals[2] / (eigenvals[0] + 1e-6)) if eigenvals[0] > 1e-6 else 0.0
    else:
        eigenvals = np.array([1.0, 1.0, 1.0])
        anisotropy = 0.0
        planarity = 0.0
    
    # Compactness
    compactness = volume_A3 / (Rg**3 + 1e-6) if Rg > 1e-6 else 0.0
    
    # Surface area estimate (approximate from mask)
    # Use binary dilation to estimate surface
    if n_voxels > 0:
        dilated = binary_dilation(component_mask, structure=np.ones((3, 3, 3)))
        surface_voxels = (dilated & ~component_mask).sum()
        surface_A2 = surface_voxels * (apix ** 2)
    else:
        surface_A2 = 0.0
    
    return {
        "volume_A3": float(volume_A3),
        "surface_A2": float(surface_A2),
        "n_pseudoatoms": int(n_points),
        "n_voxels": int(n_voxels),
        "radius_of_gyration_A": float(Rg),
        "compactness": float(compactness),
        "anisotropy": float(anisotropy),
        "planarity": float(planarity),
        "eigenval_1": float(eigenvals[0]),
        "eigenval_2": float(eigenvals[1]),
        "eigenval_3": float(eigenvals[2]),
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "com_z": float(com[2]),
    }


def compute_peak_features(
    component_mask: np.ndarray,
    ligand_map: MapVolume,
    min_separation_A: float = 2.8
) -> dict[str, float]:
    """Compute peak constellation features (for NTPs, phosphates).
    
    Args:
        component_mask: Binary mask (z, y, x) for this component
        ligand_map: MapVolume with ligand density
        min_separation_A: Minimum separation between peaks in Å
        
    Returns:
        Dictionary of peak features
    """
    # Extract density values for this component
    density_values = ligand_map.data_zyx[component_mask]
    
    if len(density_values) == 0:
        return {
            "n_peaks": 0,
            "peak_mean_spacing_A": np.nan,
            "peak_spacing_std_A": np.nan,
            "max_density": 0.0,
            "mean_density": 0.0,
        }
    
    # Find local maxima (simplified: just high-density voxels)
    # In practice, would use non-maximum suppression
    threshold = np.percentile(density_values, 75)  # Top 25% as "peaks"
    peak_mask = component_mask & (ligand_map.data_zyx >= threshold)
    
    if peak_mask.sum() == 0:
        return {
            "n_peaks": 0,
            "peak_mean_spacing_A": np.nan,
            "peak_spacing_std_A": np.nan,
            "max_density": float(density_values.max()),
            "mean_density": float(density_values.mean()),
        }
    
    # Get peak coordinates in Å
    apix = ligand_map.apix
    origin = ligand_map.origin_xyzA
    peak_voxels = np.array(np.where(peak_mask)).T  # (z, y, x)
    peak_xyz = np.zeros((len(peak_voxels), 3))
    peak_xyz[:, 0] = origin[0] + peak_voxels[:, 2] * apix  # x
    peak_xyz[:, 1] = origin[1] + peak_voxels[:, 1] * apix  # y
    peak_xyz[:, 2] = origin[2] + peak_voxels[:, 0] * apix  # z
    
    # Compute pairwise distances and find spacing
    if len(peak_xyz) > 1:
        from scipy.spatial.distance import pdist
        dists = pdist(peak_xyz)
        nearest_dists = []
        for i in range(len(peak_xyz)):
            other_dists = [np.linalg.norm(peak_xyz[i] - peak_xyz[j]) 
                          for j in range(len(peak_xyz)) if i != j]
            if other_dists:
                nearest_dists.append(min(other_dists))
        
        if nearest_dists:
            mean_spacing = np.mean(nearest_dists)
            std_spacing = np.std(nearest_dists)
        else:
            mean_spacing = np.nan
            std_spacing = np.nan
    else:
        mean_spacing = np.nan
        std_spacing = np.nan
    
    # Count linear runs (simplified: check if peaks form a line)
    linear_runs = 0
    if len(peak_xyz) >= 3:
        # Simple heuristic: if many peaks are approximately collinear
        # (would use RANSAC in practice)
        linear_runs = 1  # Placeholder
    
    return {
        "n_peaks": int(len(peak_xyz)),
        "peak_mean_spacing_A": float(mean_spacing) if not np.isnan(mean_spacing) else np.nan,
        "peak_spacing_std_A": float(std_spacing) if not np.isnan(std_spacing) else np.nan,
        "linear_runs": int(linear_runs),
        "max_density": float(density_values.max()),
        "mean_density": float(density_values.mean()),
    }


def compute_environment_features(
    component_xyz: np.ndarray,
    model_xyz: np.ndarray,
    model_df: Optional[pd.DataFrame] = None,
    radius_A: float = 6.0
) -> dict[str, float]:
    """Compute environment features from nearby protein atoms.
    
    Args:
        component_xyz: (N, 3) pseudoatom coordinates in Å
        model_xyz: (M, 3) model atom coordinates in Å
        model_df: Optional DataFrame with model atom info (resname, element, etc.)
        radius_A: Radius for environment shell in Å
        
    Returns:
        Dictionary of environment features
    """
    if len(model_xyz) == 0:
        return {
            "n_nearby_protein_atoms": 0,
            "min_protein_distance_A": np.inf,
            "mean_protein_distance_A": np.nan,
            "basic_residue_count": 0,
            "acidic_residue_count": 0,
            "polar_residue_count": 0,
            "hydrophobic_residue_count": 0,
            "has_Mg": False,
            "has_Fe": False,
            "has_Zn": False,
            "axial_HisCys": False,
        }
    
    # Component center
    component_center = component_xyz.mean(axis=0) if len(component_xyz) > 0 else np.array([0.0, 0.0, 0.0])
    
    # Build KD-tree
    tree = cKDTree(model_xyz)
    
    # Find nearby atoms
    nearby_indices = tree.query_ball_point(component_center, r=radius_A)
    nearby_indices = [idx for sublist in nearby_indices for idx in (sublist if isinstance(sublist, list) else [sublist])]
    
    n_nearby = len(nearby_indices)
    
    # Distances
    if n_nearby > 0:
        nearby_coords = model_xyz[nearby_indices]
        dists = np.linalg.norm(nearby_coords - component_center, axis=1)
        min_dist = float(dists.min())
        mean_dist = float(dists.mean())
    else:
        min_dist = np.inf
        mean_dist = np.nan
    
    # Residue chemistry (if model_df provided)
    basic_count = 0
    acidic_count = 0
    polar_count = 0
    hydrophobic_count = 0
    has_Mg = False
    has_Fe = False
    has_Zn = False
    axial_HisCys = False
    
    if model_df is not None and n_nearby > 0:
        nearby_df = model_df.iloc[nearby_indices]
        
        # Residue counts
        basic_residues = {"LYS", "ARG", "HIS"}
        acidic_residues = {"ASP", "GLU"}
        polar_residues = {"ASN", "GLN", "SER", "THR", "TYR"}
        hydrophobic_residues = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
        
        if "resname" in nearby_df.columns:
            resnames = nearby_df["resname"].str.upper()
            basic_count = resnames.isin(basic_residues).sum()
            acidic_count = resnames.isin(acidic_residues).sum()
            polar_count = resnames.isin(polar_residues).sum()
            hydrophobic_count = resnames.isin(hydrophobic_residues).sum()
            
            # Check for His/Cys (axial ligation for hemes)
            his_cys_count = (resnames.isin({"HIS", "CYS"})).sum()
            axial_HisCys = his_cys_count >= 1
        
        # Metal detection
        if "element" in nearby_df.columns:
            elements = nearby_df["element"].str.upper()
            has_Mg = (elements == "MG").any()
            has_Fe = (elements.isin({"FE", "FE2", "FE3"})).any()
            has_Zn = (elements == "ZN").any()
    
    return {
        "n_nearby_protein_atoms": int(n_nearby),
        "min_protein_distance_A": float(min_dist) if min_dist != np.inf else np.nan,
        "mean_protein_distance_A": float(mean_dist) if not np.isnan(mean_dist) else np.nan,
        "basic_residue_count": int(basic_count),
        "acidic_residue_count": int(acidic_count),
        "polar_residue_count": int(polar_count),
        "hydrophobic_residue_count": int(hydrophobic_count),
        "has_Mg": bool(has_Mg),
        "has_Fe": bool(has_Fe),
        "has_Zn": bool(has_Zn),
        "axial_HisCys": bool(axial_HisCys),
    }

