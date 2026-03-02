# crymodel/alphafold/affilter.py
"""AlphaFold model filtering and domain identification."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import gemmi
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from ..io.pdb import read_model_xyz


@dataclass
class Domain:
    """Represents a domain in the AlphaFold model."""
    domain_id: int
    residue_indices: List[int]  # Residue indices (0-based)
    chain_id: str
    start_resi: int
    end_resi: int
    ca_positions: np.ndarray  # C-alpha positions (N, 3)
    centroid: np.ndarray  # Domain centroid (3,)
    radius_of_gyration: float
    plddt_mean: float
    plddt_min: float
    plddt_max: float


@dataclass
class FilteredModel:
    """Filtered AlphaFold model with domain information."""
    structure: gemmi.Structure
    filtered_residues: List[Tuple[str, int]]  # (chain_id, resi) tuples
    removed_residues: List[Tuple[str, int]]  # Low pLDDT or filtered out
    domains: List[Domain]
    low_plddt_regions: List[Tuple[str, int, int]]  # (chain_id, start_resi, end_resi)


def get_residue_plddt(residue: gemmi.Residue) -> float:
    """Extract pLDDT from residue (average of CA atom B-factor)."""
    ca_atom = None
    for atom in residue:
        if atom.name == "CA":
            ca_atom = atom
            break
    
    if ca_atom is None:
        # Try to get any atom's B-factor
        for atom in residue:
            if atom.b_iso > 0:
                return atom.b_iso / 100.0
        return 1.0
    
    return ca_atom.b_iso / 100.0 if ca_atom.b_iso > 0 else 1.0


def detect_extended_loops(
    ca_positions: np.ndarray,
    max_ca_distance: float = 4.5,  # Max C-alpha to C-alpha distance
    min_loop_length: int = 10,  # Minimum residues for extended loop
    max_loop_length: int = 50,  # Maximum residues for extended loop
) -> List[Tuple[int, int]]:
    """Detect extended loops (barbed wire) in AlphaFold models.
    
    Args:
        ca_positions: C-alpha positions (N, 3)
        max_ca_distance: Maximum C-alpha to C-alpha distance (default 4.5 Å)
        min_loop_length: Minimum loop length to flag
        max_loop_length: Maximum loop length to flag
        
    Returns:
        List of (start_idx, end_idx) tuples for extended loops
    """
    if len(ca_positions) < 2:
        return []
    
    # Compute distances between consecutive C-alphas
    distances = np.linalg.norm(np.diff(ca_positions, axis=0), axis=1)
    
    # Find regions with consistently long distances
    extended_regions = []
    in_extended = False
    start_idx = None
    
    for i, dist in enumerate(distances):
        if dist > max_ca_distance:
            if not in_extended:
                start_idx = i
                in_extended = True
        else:
            if in_extended and start_idx is not None:
                end_idx = i + 1  # Include the residue before the gap
                loop_length = end_idx - start_idx
                if min_loop_length <= loop_length <= max_loop_length:
                    extended_regions.append((start_idx, end_idx))
                in_extended = False
                start_idx = None
    
    # Check if we end in an extended region
    if in_extended and start_idx is not None:
        end_idx = len(ca_positions)
        loop_length = end_idx - start_idx
        if min_loop_length <= loop_length <= max_loop_length:
            extended_regions.append((start_idx, end_idx))
    
    return extended_regions


def detect_low_connectivity_regions(
    ca_positions: np.ndarray,
    connectivity_threshold: float = 6.0,  # Distance threshold for connectivity
    min_neighbors: int = 2,  # Minimum neighbors within threshold
    min_region_length: int = 5,  # Minimum residues in low-connectivity region
) -> List[Tuple[int, int]]:
    """Detect regions with low local connectivity (potential artifacts).
    
    Args:
        ca_positions: C-alpha positions (N, 3)
        connectivity_threshold: Distance threshold for neighbors (Å)
        min_neighbors: Minimum neighbors required
        min_region_length: Minimum residues in region
        
    Returns:
        List of (start_idx, end_idx) tuples for low-connectivity regions
    """
    if len(ca_positions) < min_region_length:
        return []
    
    # Compute neighbor counts for each residue
    distances = cdist(ca_positions, ca_positions)
    neighbor_counts = np.sum((distances < connectivity_threshold) & (distances > 0.1), axis=1)
    
    # Find regions with consistently low neighbor counts
    low_connectivity = []
    in_region = False
    start_idx = None
    
    for i, count in enumerate(neighbor_counts):
        if count < min_neighbors:
            if not in_region:
                start_idx = i
                in_region = True
        else:
            if in_region and start_idx is not None:
                end_idx = i
                if end_idx - start_idx >= min_region_length:
                    low_connectivity.append((start_idx, end_idx))
                in_region = False
                start_idx = None
    
    # Check if we end in a low-connectivity region
    if in_region and start_idx is not None:
        end_idx = len(ca_positions)
        if end_idx - start_idx >= min_region_length:
            low_connectivity.append((start_idx, end_idx))
    
    return low_connectivity


def identify_domains(
    ca_positions: np.ndarray,
    residue_data: List[Dict],  # List of dicts with 'resi', 'plddt', etc.
    chain_id: str,
    clustering_method: str = "dbscan",
    eps: float = 15.0,  # DBSCAN eps parameter
    min_samples: int = 10,  # DBSCAN min_samples
    n_clusters: Optional[int] = None,  # For agglomerative clustering
) -> List[Domain]:
    """Identify domains using clustering analysis.
    
    Args:
        ca_positions: C-alpha positions (N, 3)
        residue_data: List of residue data dicts with 'resi', 'plddt', etc.
        chain_id: Chain identifier
        clustering_method: 'dbscan' or 'agglomerative'
        eps: DBSCAN eps parameter (Å)
        min_samples: DBSCAN min_samples parameter
        n_clusters: Number of clusters for agglomerative (None = auto)
        
    Returns:
        List of Domain objects
    """
    if len(ca_positions) < 3:
        return []
    
    # Perform clustering
    if clustering_method == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(ca_positions)
        labels = clustering.labels_
    elif clustering_method == "agglomerative":
        if n_clusters is None:
            # Auto-determine number of clusters (roughly one per 50-100 residues)
            n_clusters = max(1, len(ca_positions) // 75)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(ca_positions)
        labels = clustering.labels_
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    # Extract domains
    unique_labels = np.unique(labels)
    domains = []
    
    for domain_id, label in enumerate(unique_labels):
        if label == -1:  # DBSCAN noise points
            continue
        
        mask = labels == label
        domain_residue_data = [residue_data[i] for i in range(len(residue_data)) if mask[i]]
        domain_ca_positions = ca_positions[mask]
        domain_plddt = [res['plddt'] for res in domain_residue_data]
        domain_resi = [res['resi'] for res in domain_residue_data]
        
        if len(domain_ca_positions) < 5:  # Skip very small clusters
            continue
        
        # Compute domain properties
        centroid = np.mean(domain_ca_positions, axis=0)
        distances_to_centroid = np.linalg.norm(domain_ca_positions - centroid, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances_to_centroid**2))
        
        # Get residue range
        domain_resi_sorted = sorted(domain_resi)
        start_resi = domain_resi_sorted[0]
        end_resi = domain_resi_sorted[-1]
        
        domains.append(Domain(
            domain_id=domain_id,
            residue_indices=domain_resi,  # Actual residue numbers
            chain_id=chain_id,
            start_resi=start_resi,
            end_resi=end_resi,
            ca_positions=domain_ca_positions,
            centroid=centroid,
            radius_of_gyration=float(radius_of_gyration),
            plddt_mean=float(np.mean(domain_plddt)),
            plddt_min=float(np.min(domain_plddt)),
            plddt_max=float(np.max(domain_plddt)),
        ))
    
    return domains


def filter_alphafold_model(
    pdb_path: Path,
    plddt_threshold: float = 0.5,
    filter_extended_loops: bool = True,
    filter_low_connectivity: bool = True,
    max_ca_distance: float = 4.5,
    min_loop_length: int = 10,
    max_loop_length: int = 50,
    connectivity_threshold: float = 6.0,
    min_neighbors: int = 2,
    clustering_method: str = "dbscan",
    clustering_eps: float = 15.0,
    clustering_min_samples: int = 10,
    n_clusters: Optional[int] = None,
) -> FilteredModel:
    """Filter AlphaFold model and identify domains.
    
    Args:
        pdb_path: Path to AlphaFold PDB file
        plddt_threshold: Minimum pLDDT to keep (default 0.5)
        filter_extended_loops: If True, filter extended loops
        filter_low_connectivity: If True, filter low-connectivity regions
        max_ca_distance: Max C-alpha distance for extended loop detection
        min_loop_length: Minimum loop length to filter
        max_loop_length: Maximum loop length to filter
        connectivity_threshold: Distance threshold for connectivity (Å)
        min_neighbors: Minimum neighbors for connectivity
        clustering_method: 'dbscan' or 'agglomerative'
        clustering_eps: DBSCAN eps parameter
        clustering_min_samples: DBSCAN min_samples
        n_clusters: Number of clusters for agglomerative
        
    Returns:
        FilteredModel object
    """
    st = gemmi.read_structure(str(pdb_path))
    
    filtered_structure = gemmi.Structure()
    filtered_structure.name = st.name
    filtered_structure.spacegroup_hm = st.spacegroup_hm
    
    filtered_residues = []
    removed_residues = []
    low_plddt_regions = []
    all_domains = []
    
    for model_idx, model in enumerate(st):
        filtered_model = gemmi.Model(str(model_idx + 1))
        
        for chain in model:
            chain_id = chain.name
            filtered_chain = gemmi.Chain(chain_id)
            
            # Collect residues with their properties
            residues_data = []
            ca_positions = []
            plddt_values = []
            residue_indices = []
            
            for res_idx, res in enumerate(chain):
                plddt = get_residue_plddt(res)
                ca_atom = None
                for atom in res:
                    if atom.name == "CA":
                        ca_atom = atom
                        break
                
                if ca_atom is not None:
                    residues_data.append({
                        'residue': res,
                        'resi': res.seqid.num if res.seqid.num is not None else res_idx + 1,
                        'plddt': plddt,
                        'ca_pos': np.array([float(ca_atom.pos.x), float(ca_atom.pos.y), float(ca_atom.pos.z)]),
                    })
                    ca_positions.append(residues_data[-1]['ca_pos'])
                    plddt_values.append(plddt)
                    residue_indices.append(res_idx)
            
            if len(ca_positions) == 0:
                continue
            
            ca_positions = np.array(ca_positions)
            
            # Identify regions to filter
            regions_to_remove = set()
            
            # 1. Low pLDDT regions
            for i, plddt in enumerate(plddt_values):
                if plddt < plddt_threshold:
                    regions_to_remove.add(i)
                    # Track low pLDDT regions
                    resi = residues_data[i]['resi']
                    if not low_plddt_regions or low_plddt_regions[-1][2] != resi - 1:
                        low_plddt_regions.append((chain_id, resi, resi))
                    else:
                        # Extend last region
                        low_plddt_regions[-1] = (chain_id, low_plddt_regions[-1][1], resi)
            
            # 2. Extended loops
            if filter_extended_loops:
                extended_loops = detect_extended_loops(
                    ca_positions,
                    max_ca_distance=max_ca_distance,
                    min_loop_length=min_loop_length,
                    max_loop_length=max_loop_length,
                )
                for start_idx, end_idx in extended_loops:
                    for idx in range(start_idx, end_idx):
                        regions_to_remove.add(idx)
            
            # 3. Low connectivity regions
            if filter_low_connectivity:
                low_conn_regions = detect_low_connectivity_regions(
                    ca_positions,
                    connectivity_threshold=connectivity_threshold,
                    min_neighbors=min_neighbors,
                )
                for start_idx, end_idx in low_conn_regions:
                    for idx in range(start_idx, end_idx):
                        regions_to_remove.add(idx)
            
            # Filter residues
            filtered_ca_positions = []
            filtered_residue_data = []
            
            for i, res_data in enumerate(residues_data):
                if i not in regions_to_remove:
                    # Keep residue - need to create a new residue and copy atoms
                    # because the original residue belongs to another chain
                    original_res = res_data['residue']
                    new_res = gemmi.Residue()
                    new_res.name = original_res.name
                    # Copy seqid properly - SeqId takes (int, str) or (str)
                    if original_res.seqid.num is not None:
                        icode = original_res.seqid.icode if hasattr(original_res.seqid, 'icode') else ""
                        new_res.seqid = gemmi.SeqId(int(original_res.seqid.num), str(icode))
                    else:
                        new_res.seqid = gemmi.SeqId(str(original_res.seqid))
                    if hasattr(original_res, 'subchain') and original_res.subchain:
                        new_res.subchain = original_res.subchain
                    
                    # Copy all atoms from original residue
                    for atom in original_res:
                        new_atom = gemmi.Atom()
                        new_atom.name = atom.name
                        new_atom.element = atom.element
                        new_atom.pos = gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)
                        new_atom.occ = atom.occ
                        new_atom.b_iso = atom.b_iso
                        if hasattr(atom, 'serial'):
                            new_atom.serial = atom.serial
                        new_res.add_atom(new_atom)
                    
                    filtered_chain.add_residue(new_res)
                    filtered_residues.append((chain_id, res_data['resi']))
                    filtered_ca_positions.append(res_data['ca_pos'])
                    filtered_residue_data.append({
                        'resi': res_data['resi'],
                        'plddt': res_data['plddt'],
                    })
                else:
                    removed_residues.append((chain_id, res_data['resi']))
            
            if len(filtered_chain) > 0:
                filtered_model.add_chain(filtered_chain)
                
                # Identify domains in filtered chain
                if len(filtered_ca_positions) >= 5:
                    filtered_ca_positions = np.array(filtered_ca_positions)
                    domains = identify_domains(
                        filtered_ca_positions,
                        filtered_residue_data,
                        chain_id,
                        clustering_method=clustering_method,
                        eps=clustering_eps,
                        min_samples=clustering_min_samples,
                        n_clusters=n_clusters,
                    )
                    all_domains.extend(domains)
        
        if len(filtered_model) > 0:
            filtered_structure.add_model(filtered_model)
    
    return FilteredModel(
        structure=filtered_structure,
        filtered_residues=filtered_residues,
        removed_residues=removed_residues,
        domains=all_domains,
        low_plddt_regions=low_plddt_regions,
    )

