# crymodel/domains/pdbcom.py
"""Compute domain centers of mass and output as PDB."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import csv
import numpy as np
import gemmi

# Element masses (atomic mass units)
ELEMENT_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'P': 30.974, 'S': 32.066, 'FE': 55.845, 'ZN': 65.38,
    'MG': 24.305, 'CA': 40.078, 'K': 39.098, 'NA': 22.990,
}


def parse_domain_spec(domains_json: Path) -> Dict[str, Dict[str, str]]:
    """Parse domain specification from JSON file.
    
    Format:
    {
      "LBD-S1": {"A": "45-125"},
      "LBD-S2": {"A": "260-330"}
    }
    """
    with open(domains_json, 'r') as f:
        return json.load(f)


def parse_residue_range(range_str: str) -> Tuple[int, int]:
    """Parse residue range string like "45-125" or "45"."""
    if '-' in range_str:
        start, end = range_str.split('-')
        return int(start.strip()), int(end.strip())
    else:
        resnum = int(range_str.strip())
        return resnum, resnum


def get_atoms_in_range(
    structure: gemmi.Structure,
    chain_id: str,
    start_res: int,
    end_res: int,
    atom_filter: str = "all",
) -> List[gemmi.Atom]:
    """Get atoms in residue range.
    
    Args:
        structure: Gemmi structure
        chain_id: Chain identifier
        start_res: Start residue number
        end_res: End residue number
        atom_filter: "all", "backbone", or "CA"
    """
    atoms = []
    
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            for residue in chain:
                resnum = residue.seqid.num
                if start_res <= resnum <= end_res:
                    for atom in residue:
                        if atom_filter == "CA" and atom.name != "CA":
                            continue
                        elif atom_filter == "backbone":
                            if atom.name not in ["N", "CA", "C", "O"]:
                                continue
                        atoms.append(atom)
    
    return atoms


def compute_com(
    atoms: List[gemmi.Atom],
    mass_weighted: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """Compute center of mass.
    
    Returns:
        (com_position, total_mass, num_atoms)
    """
    if len(atoms) == 0:
        return np.zeros(3), 0.0, 0
    
    positions = []
    masses = []
    
    for atom in atoms:
        pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        positions.append(pos)
        
        if mass_weighted:
            element = atom.element.name.upper()
            mass = ELEMENT_MASSES.get(element, 12.011)  # Default to C
            masses.append(mass)
        else:
            masses.append(1.0)
    
    positions = np.array(positions)
    masses = np.array(masses)
    
    total_mass = masses.sum()
    if total_mass > 0:
        com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    else:
        com = positions.mean(axis=0)
    
    return com, float(total_mass), len(atoms)


def compute_domain_coms(
    structure: gemmi.Structure,
    domains: Dict[str, Dict[str, str]],
    mass_weighted: bool = True,
    atom_filter: str = "all",
) -> Dict[str, Dict]:
    """Compute centers of mass for all domains.
    
    Returns:
        Dictionary mapping domain name to {
            'com': [x, y, z],
            'mass': total_mass,
            'num_atoms': num_atoms,
            'chain': chain_id,
        }
    """
    results = {}
    
    for domain_name, chain_ranges in domains.items():
        all_atoms = []
        chain_ids = []
        
        for chain_id, range_str in chain_ranges.items():
            start_res, end_res = parse_residue_range(range_str)
            atoms = get_atoms_in_range(structure, chain_id, start_res, end_res, atom_filter)
            all_atoms.extend(atoms)
            chain_ids.append(chain_id)
        
        if len(all_atoms) == 0:
            continue
        
        com, total_mass, num_atoms = compute_com(all_atoms, mass_weighted)
        
        results[domain_name] = {
            'com': com.tolist(),
            'mass': total_mass,
            'num_atoms': num_atoms,
            'chains': chain_ids,
        }
    
    return results


def write_domain_com_pdb(
    domain_coms: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Write domain COMs as PDB file."""
    lines = []
    serial = 1
    
    for domain_name, data in domain_coms.items():
        x, y, z = data['com']
        # Use first chain or 'X' as default
        chain_id = data['chains'][0] if data['chains'] else 'X'
        
        # Format PDB ATOM record
        lines.append(
            f"HETATM{serial:5d}  C  DOM {chain_id:1s}{serial:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
        )
        serial += 1
    
    with open(output_path, 'w') as f:
        f.writelines(lines)


def write_domain_com_csv(
    domain_coms: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Write domain COMs as CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'x', 'y', 'z', 'num_atoms', 'mass', 'chains'])
        
        for domain_name, data in domain_coms.items():
            x, y, z = data['com']
            chains_str = ','.join(data['chains'])
            writer.writerow([
                domain_name,
                f"{x:.3f}",
                f"{y:.3f}",
                f"{z:.3f}",
                data['num_atoms'],
                f"{data['mass']:.2f}",
                chains_str,
            ])

