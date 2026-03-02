# crymodel/nucleotide/templates.py
"""Template management for BaseHunter nucleotide classification."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import gemmi
from scipy.ndimage import gaussian_filter

from ..io.mrc import read_map, MapVolume
from ..io.pdb import read_model_xyz


class TemplateLibrary:
    """
    Manages purine and pyrimidine templates for nucleotide classification.
    
    Supports:
    - PDB models (auto-converted to density at target resolution)
    - MRC density maps (at various resolutions, auto-selected)
    - Multiple templates per class (averaged or best-match)
    """
    
    def __init__(
        self,
        template_dir: Optional[Path] = None,
        purine_templates: Optional[List[Union[str, Path]]] = None,
        pyrimidine_templates: Optional[List[Union[str, Path]]] = None,
    ):
        """
        Initialize template library.
        
        Args:
            template_dir: Directory containing templates (see directory structure below)
            purine_templates: List of template paths (PDB or MRC) for purines
            pyrimidine_templates: List of template paths (PDB or MRC) for pyrimidines
            
        Directory structure (if using template_dir):
            templates/
                purine/
                    template1.pdb (or .mrc)
                    template2.pdb (or .mrc)
                    ...
                pyrimidine/
                    template1.pdb (or .mrc)
                    template2.pdb (or .mrc)
                    ...
        """
        self.purine_templates: List[Path] = []
        self.pyrimidine_templates: List[Path] = []
        self.purine_resolutions: List[Optional[float]] = []  # None for PDB, resolution for MRC
        self.pyrimidine_resolutions: List[Optional[float]] = []
        
        if template_dir:
            self._load_from_directory(Path(template_dir))
        elif purine_templates or pyrimidine_templates:
            if purine_templates:
                self.purine_templates = [Path(p) for p in purine_templates]
                self.purine_resolutions = [self._get_resolution(p) for p in self.purine_templates]
            if pyrimidine_templates:
                self.pyrimidine_templates = [Path(p) for p in pyrimidine_templates]
                self.pyrimidine_resolutions = [self._get_resolution(p) for p in self.pyrimidine_templates]
        else:
            raise ValueError("Must provide either template_dir or explicit template lists")
    
    def _load_from_directory(self, template_dir: Path):
        """Load templates from directory structure.
        
        Supports two formats:
        1. Subdirectory structure: template_dir/purine/ and template_dir/pyrimidine/
        2. Flat structure with naming: template-{base}-{resolution}.mrc
           where base is a/g (purine) or c/t (pyrimidine)
           and resolution is x10 (e.g., 35 = 3.5 Å)
        """
        template_dir = Path(template_dir)
        
        # Try subdirectory structure first
        purine_dir = template_dir / "purine"
        pyrimidine_dir = template_dir / "pyrimidine"
        
        if purine_dir.exists() and pyrimidine_dir.exists():
            # Subdirectory structure
            for template_file in purine_dir.glob("*.pdb"):
                self.purine_templates.append(template_file)
                self.purine_resolutions.append(None)
            for template_file in purine_dir.glob("*.mrc"):
                self.purine_templates.append(template_file)
                self.purine_resolutions.append(self._get_resolution(template_file))
            
            for template_file in pyrimidine_dir.glob("*.pdb"):
                self.pyrimidine_templates.append(template_file)
                self.pyrimidine_resolutions.append(None)
            for template_file in pyrimidine_dir.glob("*.mrc"):
                self.pyrimidine_templates.append(template_file)
                self.pyrimidine_resolutions.append(self._get_resolution(template_file))
        else:
            # Flat structure with naming convention: template-{base}-{resolution}.mrc
            # Purines: A, G
            # Pyrimidines: C, T
            for template_file in template_dir.glob("template-*.mrc"):
                filename = template_file.stem.lower()  # e.g., "template-a-35"
                parts = filename.split("-")
                if len(parts) >= 3:
                    base = parts[1].lower()  # a, g, c, t
                    res_str = parts[2]  # e.g., "35" for 3.5 Å
                    
                    # Parse resolution (x10 format)
                    try:
                        resolution = float(res_str) / 10.0  # 35 -> 3.5 Å
                    except ValueError:
                        resolution = None
                    
                    if base in ['a', 'g']:
                        # Purine
                        self.purine_templates.append(template_file)
                        self.purine_resolutions.append(resolution)
                    elif base in ['c', 't']:
                        # Pyrimidine
                        self.pyrimidine_templates.append(template_file)
                        self.pyrimidine_resolutions.append(resolution)
            
            # Also check for PDB files with same naming
            for template_file in template_dir.glob("template-*.pdb"):
                filename = template_file.stem.lower()
                parts = filename.split("-")
                if len(parts) >= 2:
                    base = parts[1].lower()
                    if base in ['a', 'g']:
                        self.purine_templates.append(template_file)
                        self.purine_resolutions.append(None)
                    elif base in ['c', 't']:
                        self.pyrimidine_templates.append(template_file)
                        self.pyrimidine_resolutions.append(None)
        
        if not self.purine_templates:
            raise ValueError(f"No purine templates found in {template_dir}")
        if not self.pyrimidine_templates:
            raise ValueError(f"No pyrimidine templates found in {template_dir}")
    
    def _get_resolution(self, template_path: Path) -> Optional[float]:
        """Get resolution from MRC file header, or None for PDB."""
        if template_path.suffix.lower() in ['.mrc', '.map', '.ccp4']:
            try:
                mv = read_map(str(template_path))
                # Try to get resolution from header (if available)
                # For now, return None and we'll use the target resolution
                return None  # Could parse from header if available
            except:
                return None
        return None  # PDB files
    
    def get_templates_at_resolution(
        self,
        target_resolution: float,
        target_apix: float,
        target_shape: Tuple[int, int, int],
        target_origin: np.ndarray,
        use_average: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get purine and pyrimidine templates at target resolution.
        
        Args:
            target_resolution: Target resolution in Å
            target_apix: Target voxel size in Å
            target_shape: Target map shape (z, y, x)
            target_origin: Target map origin in Å (x, y, z)
            use_average: If True, average multiple templates; if False, use best match
            
        Returns:
            (purine_template, pyrimidine_template) as density maps
        """
        purine_template = self._get_class_template(
            self.purine_templates,
            self.purine_resolutions,
            target_resolution,
            target_apix,
            target_shape,
            target_origin,
            use_average,
        )
        
        pyrimidine_template = self._get_class_template(
            self.pyrimidine_templates,
            self.pyrimidine_resolutions,
            target_resolution,
            target_apix,
            target_shape,
            target_origin,
            use_average,
        )
        
        return purine_template, pyrimidine_template
    
    def _get_class_template(
        self,
        template_paths: List[Path],
        template_resolutions: List[Optional[float]],
        target_resolution: float,
        target_apix: float,
        target_shape: Tuple[int, int, int],
        target_origin: np.ndarray,
        use_average: bool,
    ) -> np.ndarray:
        """Get template for one class (purine or pyrimidine)."""
        if not template_paths:
            raise ValueError("No templates available")
        
        template_maps = []
        
        for template_path, template_res in zip(template_paths, template_resolutions):
            if template_path.suffix.lower() in ['.pdb', '.cif']:
                # Convert PDB to density at target resolution
                template_map = self._pdb_to_density(
                    template_path,
                    target_apix,
                    target_shape,
                    target_origin,
                    target_resolution,
                )
            else:
                # Load MRC and resample to target resolution if needed
                template_map = self._load_and_resample_mrc(
                    template_path,
                    target_resolution,
                    target_apix,
                    target_shape,
                    target_origin,
                )
            template_maps.append(template_map)
        
        if use_average and len(template_maps) > 1:
            # Average all templates
            return np.mean(template_maps, axis=0).astype(np.float32)
        else:
            # Use first template (or could select best match based on resolution)
            return template_maps[0]
    
    def _pdb_to_density(
        self,
        pdb_path: Path,
        target_apix: float,
        target_shape: Tuple[int, int, int],
        target_origin: np.ndarray,
        resolution_A: float,
    ) -> np.ndarray:
        """Convert PDB structure to density map (similar to foldhunter)."""
        st = gemmi.read_structure(str(pdb_path))
        
        # Collect atom positions
        atoms_xyz = []
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        # Skip hydrogens
                        element_name = atom.element.name if atom.element else atom.name.strip()[0] if atom.name.strip() else "C"
                        if element_name.upper() == "H":
                            continue
                        
                        pos = atom.pos
                        atoms_xyz.append([float(pos.x), float(pos.y), float(pos.z)])
        
        if not atoms_xyz:
            return np.zeros(target_shape, dtype=np.float32)
        
        atoms_xyz = np.array(atoms_xyz, dtype=np.float32)
        
        # Create empty map
        density = np.zeros(target_shape, dtype=np.float32)
        z_shape, y_shape, x_shape = target_shape
        
        # Place atoms as Gaussian spheres
        atom_radius_A = 1.5
        radius_vox = atom_radius_A / target_apix
        sigma_vox = radius_vox / 2.0
        
        kernel_radius = int(np.ceil(3 * sigma_vox))
        if kernel_radius > 0:
            k_range = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
            KZ, KY, KX = np.meshgrid(k_range, k_range, k_range, indexing='ij')
            kernel = np.exp(-(KZ**2 + KY**2 + KX**2) / (2 * sigma_vox**2))
            kernel /= np.sum(kernel)
        
        # Place atoms
        for x, y, z in atoms_xyz:
            vx = (x - target_origin[0]) / target_apix
            vy = (y - target_origin[1]) / target_apix
            vz = (z - target_origin[2]) / target_apix
            
            vx_int = int(np.round(vx))
            vy_int = int(np.round(vy))
            vz_int = int(np.round(vz))
            
            z0 = max(0, vz_int - kernel_radius)
            z1 = min(z_shape, vz_int + kernel_radius + 1)
            y0 = max(0, vy_int - kernel_radius)
            y1 = min(y_shape, vy_int + kernel_radius + 1)
            x0 = max(0, vx_int - kernel_radius)
            x1 = min(x_shape, vx_int + kernel_radius + 1)
            
            kz0 = kernel_radius - (vz_int - z0)
            kz1 = kernel_radius + (z1 - vz_int)
            ky0 = kernel_radius - (vy_int - y0)
            ky1 = kernel_radius + (y1 - vy_int)
            kx0 = kernel_radius - (vx_int - x0)
            kx1 = kernel_radius + (x1 - vx_int)
            
            if z0 < z1 and y0 < y1 and x0 < x1:
                density[z0:z1, y0:y1, x0:x1] += kernel[kz0:kz1, ky0:ky1, kx0:kx1]
        
        # Apply Gaussian blur to simulate resolution
        resolution_sigma_vox = (0.939 * resolution_A) / (2.355 * target_apix)
        if resolution_sigma_vox > 0.5:
            density = gaussian_filter(density, sigma=resolution_sigma_vox, mode='constant')
        
        return density.astype(np.float32)
    
    def _load_and_resample_mrc(
        self,
        mrc_path: Path,
        target_resolution: float,
        target_apix: float,
        target_shape: Tuple[int, int, int],
        target_origin: np.ndarray,
    ) -> np.ndarray:
        """Load MRC and resample to target resolution/shape if needed."""
        mv = read_map(str(mrc_path))
        
        # If shapes match and apix is close, use directly
        if (mv.data_zyx.shape == target_shape and 
            abs(mv.apix - target_apix) < 0.01):
            return mv.data_zyx.astype(np.float32)
        
        # Otherwise, need to resample
        # For now, simple approach: if resolution is close, just resample spatially
        # More sophisticated: could apply resolution-dependent blur
        from scipy.ndimage import zoom
        
        # Compute zoom factors
        zoom_z = target_shape[0] / mv.data_zyx.shape[0]
        zoom_y = target_shape[1] / mv.data_zyx.shape[1]
        zoom_x = target_shape[2] / mv.data_zyx.shape[2]
        
        resampled = zoom(mv.data_zyx, (zoom_z, zoom_y, zoom_x), order=1, mode='constant', cval=0.0)
        
        # If template resolution is different, apply additional blur
        # (This is simplified - could be more sophisticated)
        if target_resolution > 2.5:  # Lower resolution
            # Apply additional blur
            extra_sigma = (target_resolution - 2.5) / (2.355 * target_apix)
            if extra_sigma > 0.1:
                resampled = gaussian_filter(resampled, sigma=extra_sigma, mode='constant')
        
        return resampled.astype(np.float32)

