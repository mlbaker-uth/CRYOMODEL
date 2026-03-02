# crymodel/prep/fitprep.py
"""Preflight checker for maps & models."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np
import gemmi

from ..io.mrc import read_map, MapVolume
from ..validation.local_cc import compute_local_cc_variants


def check_voxel_grid(map_vol: MapVolume) -> Dict:
    """Check voxel size and grid consistency."""
    checks = {
        'voxel_size': float(map_vol.apix),
        'grid_shape': list(map_vol.data_zyx.shape),
        'origin': list(map_vol.origin_xyzA),
        'warnings': [],
    }
    
    # Check for non-orthogonal (would need additional header info)
    # Check voxel size consistency
    if map_vol.apix <= 0:
        checks['warnings'].append("Invalid voxel size")
    
    return checks


def check_origin_alignment(
    map_vol: MapVolume,
    structure: gemmi.Structure,
) -> Dict:
    """Check alignment between map origin and model centroid."""
    # Compute model centroid
    positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    positions.append([atom.pos.x, atom.pos.y, atom.pos.z])
    
    if len(positions) == 0:
        return {'centroid': [0, 0, 0], 'offset': [0, 0, 0], 'warnings': ['No atoms in model']}
    
    positions = np.array(positions)
    centroid = positions.mean(axis=0)
    
    # Compute offset
    origin = np.array(map_vol.origin_xyzA)
    offset = centroid - origin
    
    # Suggest integer-voxel shift
    apix = map_vol.apix
    suggested_shift = (offset / apix).round() * apix
    
    checks = {
        'model_centroid': centroid.tolist(),
        'map_origin': origin.tolist(),
        'offset': offset.tolist(),
        'suggested_shift': suggested_shift.tolist(),
        'warnings': [],
    }
    
    # Warn if offset is large
    if np.linalg.norm(offset) > 10.0:
        checks['warnings'].append(f"Large offset between model and map: {np.linalg.norm(offset):.2f} Å")
    
    return checks


def check_intensity_normalization(map_vol: MapVolume) -> Dict:
    """Check map intensity normalization."""
    data = map_vol.data_zyx
    mean = float(data.mean())
    std = float(data.std())
    
    checks = {
        'mean': mean,
        'std': std,
        'min': float(data.min()),
        'max': float(data.max()),
        'warnings': [],
    }
    
    # Suggest normalization
    if abs(mean) > 0.1 or abs(std - 1.0) > 0.1:
        checks['warnings'].append("Map may benefit from normalization (zero mean, unit variance)")
        checks['suggested_normalization'] = {
            'subtract': mean,
            'divide': std,
        }
    
    return checks


def check_quick_fit(
    map_vol: MapVolume,
    structure: gemmi.Structure,
) -> Dict:
    """Quick fit metric using CC around Cαs."""
    # Get Cα positions
    ca_positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                ca = residue.find('CA')
                if ca:
                    ca_positions.append([ca.pos.x, ca.pos.y, ca.pos.z])
    
    if len(ca_positions) == 0:
        return {'cc_mask': 0.0, 'warnings': ['No Cα atoms found']}
    
    ca_positions = np.array(ca_positions)
    
    # Compute CC_mask
    cc_features = compute_local_cc_variants(ca_positions, map_vol)
    
    checks = {
        'cc_mask': cc_features['CC_mask'],
        'cc_box': cc_features['CC_box'],
        'zncc': cc_features['ZNCC'],
        'warnings': [],
    }
    
    if cc_features['CC_mask'] < 0.3:
        checks['warnings'].append("Low CC_mask - model may not fit map well")
    
    return checks


def check_map_model_alignment(
    map_path: str,
    model_path: str,
    half1_path: Optional[str] = None,
    half2_path: Optional[str] = None,
) -> Dict:
    """Run all preflight checks."""
    # Load map
    map_vol = read_map(map_path)
    
    # Load model
    structure = gemmi.read_structure(model_path)
    
    # Run checks
    results = {
        'map_file': map_path,
        'model_file': model_path,
        'voxel_grid': check_voxel_grid(map_vol),
        'origin_alignment': check_origin_alignment(map_vol, structure),
        'intensity_normalization': check_intensity_normalization(map_vol),
        'quick_fit': check_quick_fit(map_vol, structure),
    }
    
    # Collect all warnings
    all_warnings = []
    for check_name, check_data in results.items():
        if isinstance(check_data, dict) and 'warnings' in check_data:
            all_warnings.extend([f"{check_name}: {w}" for w in check_data['warnings']])
    
    results['all_warnings'] = all_warnings
    
    return results


def generate_report(results: Dict, output_path: Path) -> None:
    """Generate JSON report."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def apply_fixes(
    map_vol: MapVolume,
    structure: gemmi.Structure,
    fixes: Dict,
) -> tuple:
    """Apply suggested fixes to map and model."""
    from ..io.mrc import write_map
    
    fixed_map = map_vol
    fixed_structure = gemmi.Structure(structure)
    
    # Apply origin shift if requested
    if 'origin_shift' in fixes:
        shift = np.array(fixes['origin_shift'])
        # Shift model
        for model in fixed_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.pos.x += shift[0]
                        atom.pos.y += shift[1]
                        atom.pos.z += shift[2]
    
    # Apply normalization if requested
    if 'normalize' in fixes and fixes['normalize']:
        data = fixed_map.data_zyx.copy()
        mean = data.mean()
        std = data.std()
        if std > 0:
            data = (data - mean) / std
        fixed_map.data_zyx = data
    
    return fixed_map, fixed_structure

