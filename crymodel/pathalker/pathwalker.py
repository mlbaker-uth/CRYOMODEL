# crymodel/pathalker/pathwalker.py
"""Main pathwalking engine."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Literal

from ..io.mrc import MapVolume, read_map
from .pseudoatoms import generate_pseudoatoms, add_noise_to_pseudoatoms, PseudoatomMethod
from .distances import compute_distance_matrix, prepare_tsp_distance_matrix
from .tsp_solver import solve_tsp_ortools, solve_tsp_lkh
from .path_evaluation import evaluate_path, calculate_path_statistics
from ..io.site_export import _pdb_atom_line


def pathwalk(
    map_vol: MapVolume,
    threshold: float,
    n_pseudoatoms: int,
    pseudoatom_method: PseudoatomMethod = "kmeans",
    map_weighted: bool = False,
    tsp_solver: Literal["ortools", "lkh"] = "ortools",
    time_limit_seconds: int = 30,
    noise_level: float = 0.0,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int], float]:
    """Run pathwalking on a density map.
    
    Args:
        map_vol: MapVolume with density data
        threshold: Density threshold for pseudoatom generation
        n_pseudoatoms: Number of pseudoatoms (should match C-alpha count)
        pseudoatom_method: Method for generating pseudoatoms
        map_weighted: If True, use map-weighted distances
        tsp_solver: TSP solver to use ('ortools' or 'lkh')
        time_limit_seconds: Time limit for TSP solver
        noise_level: Noise level to add to pseudoatoms (Å)
        random_state: Random seed
        verbose: Print progress information
        
    Returns:
        Tuple of (path coordinates in Å, route as node indices, path length in Å)
    """
    # Generate pseudoatoms
    if verbose:
        print(f"Generating {n_pseudoatoms} pseudoatoms at threshold {threshold}...")
    
    pseudoatoms = generate_pseudoatoms(
        map_vol,
        threshold=threshold,
        n_pseudoatoms=n_pseudoatoms,
        method=pseudoatom_method,
        random_state=random_state,
    )
    
    # Add noise if requested
    if noise_level > 0:
        pseudoatoms = add_noise_to_pseudoatoms(pseudoatoms, noise_level, random_state)
    
    # Compute distance matrix
    if verbose:
        print("Computing distance matrix...")
    
    distance_matrix = compute_distance_matrix(
        pseudoatoms,
        map_vol=map_vol if map_weighted else None,
        map_weighted=map_weighted,
        threshold=threshold,
    )
    
    # Prepare for TSP solver
    tsp_distance_matrix = prepare_tsp_distance_matrix(distance_matrix, add_depot=True)
    
    # Solve TSP
    if verbose:
        print(f"Solving TSP using {tsp_solver}...")
    
    if tsp_solver == "ortools":
        route, path_length = solve_tsp_ortools(
            tsp_distance_matrix,
            time_limit_seconds=time_limit_seconds,
            verbose=verbose,
        )
    elif tsp_solver == "lkh":
        route, path_length = solve_tsp_lkh(
            tsp_distance_matrix,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown TSP solver: {tsp_solver}")
    
    # Handle depot in route (depot is index 0, pseudoatoms are 1..n)
    # Remove depot nodes and adjust indices
    route_no_depot = [r - 1 for r in route if r > 0]  # Subtract 1 to get pseudoatom indices (0..n-1)
    
    # Remove duplicates while preserving order
    seen = set()
    route_clean = []
    for r in route_no_depot:
        if r not in seen and 0 <= r < len(pseudoatoms):
            route_clean.append(r)
            seen.add(r)
    
    if len(route_clean) != len(pseudoatoms):
        if verbose:
            print(f"  Warning: Route has {len(route_clean)} nodes, expected {len(pseudoatoms)}")
        # If route is incomplete, add missing nodes at the end
        missing = set(range(len(pseudoatoms))) - set(route_clean)
        route_clean.extend(sorted(missing))
    
    # Reorder pseudoatoms according to route
    path_coordinates = pseudoatoms[route_clean]
    
    # Evaluate path
    if verbose:
        evaluate_path(path_coordinates, verbose=True)
    
    return path_coordinates, route, path_length


def write_path_pdb(
    path_coordinates: np.ndarray,
    output_path: Path,
    chain_id: str = "A",
    resname: str = "GLY",
    atom_name: str = "CA",
) -> None:
    """Write path coordinates to PDB file.
    
    Args:
        path_coordinates: (N, 3) array of path coordinates in Å (x, y, z)
        output_path: Output PDB file path
        chain_id: Chain ID for PDB file
        resname: Residue name (default: GLY for C-alpha)
        atom_name: Atom name (default: CA for C-alpha)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1                      \n")
        
        for i, (x, y, z) in enumerate(path_coordinates, start=1):
            line = _pdb_atom_line(
                serial=i,
                name=atom_name,
                resname=resname,
                chain=chain_id,
                resi=i,
                x=float(x),
                y=float(y),
                z=float(z),
                element="C",
            )
            f.write(line + "\n")
        
        f.write("TER\nEND\n")


def write_path_pdb_with_probabilities(
    path_coordinates: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    chain_id: str = "A",
    resname: str = "GLY",
    atom_name: str = "CA",
) -> None:
    """Write path coordinates to PDB file with probabilities in B-factor column.
    
    Args:
        path_coordinates: (N, 3) array of path coordinates in Å
        probabilities: (N,) array of probabilities (0-1)
        output_path: Output PDB file path
        chain_id: Chain ID for PDB file
        resname: Residue name
        atom_name: Atom name
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Scale probabilities to reasonable B-factor range (0-100)
    bfactors = probabilities * 100.0
    
    with open(output_path, "w") as f:
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1                      \n")
        
        for i, ((x, y, z), bfactor) in enumerate(zip(path_coordinates, bfactors), start=1):
            # Format PDB line with B-factor
            line = (
                f"ATOM  {i:5d} {atom_name:>4s} {resname:>3s} {chain_id:1s}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:5.2f}           C"
            )
            f.write(line + "\n")
        
        f.write("TER\nEND\n")

