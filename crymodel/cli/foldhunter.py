# crymodel/cli/foldhunter.py
"""CLI for FoldHunter: exhaustive cross-correlation search."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
import numpy as np
import gemmi

from ..fitting.foldhunter import foldhunter_search, apply_transformation
from ..io.mrc import write_map
from ..io.pdb import read_model_xyz

app = typer.Typer(no_args_is_help=True)


@app.command()
def search(
    target_map: Path = typer.Argument(..., help="Target density map (.mrc/.map)"),
    probe_pdb: Optional[Path] = typer.Option(None, "--probe-pdb", help="Probe PDB file (AlphaFold model)"),
    probe_map: Optional[Path] = typer.Option(None, "--probe-map", help="Probe density map (.mrc/.map)"),
    resolution: float = typer.Option(3.0, "--resolution", help="Resolution for probe map generation (Å)"),
    plddt_threshold: float = typer.Option(0.5, "--plddt-threshold", help="Minimum pLDDT to include atoms (0-1)"),
    density_threshold: Optional[float] = typer.Option(None, "--density-threshold", help="Density threshold for inclusion (None = dynamic)"),
    coarse_angle_step: Optional[float] = typer.Option(None, "--coarse-angle", help="Coarse rotation step (degrees, auto if None)"),
    fine_angle_step: float = typer.Option(1.0, "--fine-angle", help="Fine rotation step (degrees, default 1.0)"),
    coarse_translation_step: float = typer.Option(5.0, "--coarse-translation", help="Coarse translation step (Å)"),
    fine_translation_step: float = typer.Option(1.0, "--fine-translation", help="Fine translation step (Å)"),
    n_coarse_rotations: Optional[int] = typer.Option(None, "--n-coarse", help="Number of rotations for coarse search (auto if None)"),
    symmetry: Optional[int] = typer.Option(None, "--symmetry", help="Rotational symmetry (e.g., 3 for C3)"),
    out_of_bounds_penalty: float = typer.Option(0.1, "--out-of-bounds-penalty", help="Weight for out-of-bounds penalty"),
    top_n: int = typer.Option(10, "--top-n", help="Number of top candidates to return"),
    peak_width_deg: float = typer.Option(10.0, "--peak-width", help="Angular width for peak clustering (degrees)"),
    max_peaks_coarse: int = typer.Option(20, "--max-peaks-coarse", help="Maximum peaks from coarse round"),
    out_dir: Path = typer.Option(Path("foldhunter_outputs"), "--out-dir", help="Output directory"),
):
    """Run FoldHunter exhaustive cross-correlation search.
    
    Finds optimal fit of AlphaFold model (or probe map) to target density map
    using FFT-based cross-correlation with coarse-to-fine search strategy.
    """
    if probe_pdb is None and probe_map is None:
        typer.echo("Error: Must provide either --probe-pdb or --probe-map", err=True)
        raise typer.Exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Running FoldHunter search...")
    typer.echo(f"  Target map: {target_map}")
    if probe_pdb:
        typer.echo(f"  Probe PDB: {probe_pdb}")
    if probe_map:
        typer.echo(f"  Probe map: {probe_map}")
    typer.echo(f"  Resolution: {resolution} Å")
    typer.echo(f"  pLDDT threshold: {plddt_threshold}")
    
    results = foldhunter_search(
        probe_pdb=probe_pdb,
        probe_map=probe_map,
        target_map=target_map,
        resolution_A=resolution,
        plddt_threshold=plddt_threshold,
        density_threshold=density_threshold,
        coarse_angle_step=coarse_angle_step,
        fine_angle_step=fine_angle_step,
        coarse_translation_step=coarse_translation_step,
        fine_translation_step=fine_translation_step,
        n_coarse_rotations=n_coarse_rotations,
        symmetry=symmetry,
        out_of_bounds_penalty_weight=out_of_bounds_penalty,
        top_n_candidates=top_n,
        peak_width_deg=peak_width_deg,
        max_peaks_coarse=max_peaks_coarse,
    )
    
    typer.echo(f"\nFound {len(results)} candidate fits:")
    
    # Write results
    results_csv = out_dir / "foldhunter_results.csv"
    with open(results_csv, 'w') as f:
        f.write("rank,correlation,atom_inclusion_score,n_atoms_in_density,n_atoms_total,out_of_bounds_penalty,"
                "translation_x,translation_y,translation_z,rotation_w,rotation_x,rotation_y,rotation_z\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i},{result.correlation:.6f},{result.atom_inclusion_score:.4f},"
                   f"{result.n_atoms_in_density},{result.n_atoms_total},{result.out_of_bounds_penalty:.6f},"
                   f"{result.translation[0]:.3f},{result.translation[1]:.3f},{result.translation[2]:.3f},"
                   f"{result.rotation[0]:.6f},{result.rotation[1]:.6f},{result.rotation[2]:.6f},{result.rotation[3]:.6f}\n")
            typer.echo(f"  {i}. Correlation: {result.correlation:.4f}, "
                      f"Inclusion: {result.atom_inclusion_score:.2%}, "
                      f"Atoms in density: {result.n_atoms_in_density}/{result.n_atoms_total}")
    
    typer.echo(f"\nResults written to {results_csv}")
    
    # Write top candidate as transformed PDB if probe was PDB
    if probe_pdb and len(results) > 0:
        top_result = results[0]
        atoms_xyz = read_model_xyz(str(probe_pdb), remove_hydrogens=True)
        transformed_atoms = apply_transformation(atoms_xyz, top_result.rotation, top_result.translation)
        
        # Write transformed PDB
        output_pdb = out_dir / "foldhunter_top_fit.pdb"
        st = gemmi.read_structure(str(probe_pdb))
        
        # Apply transformation to structure
        from ..fitting.foldhunter import quaternion_to_rotation_matrix
        R = quaternion_to_rotation_matrix(top_result.rotation)
        t = top_result.translation
        
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        pos = atom.pos
                        pos_xyz = np.array([float(pos.x), float(pos.y), float(pos.z)])
                        pos_rotated = pos_xyz @ R.T
                        pos_translated = pos_rotated + t
                        atom.pos = gemmi.Position(pos_translated[0], pos_translated[1], pos_translated[2])
        
        st.write_pdb(str(output_pdb))
        typer.echo(f"Top fit written to {output_pdb}")
    
    typer.echo(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    app()

