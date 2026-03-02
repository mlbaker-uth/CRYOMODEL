# crymodel/cli/affilter.py
"""CLI for AlphaFold model filtering and domain identification."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
import pandas as pd

from ..alphafold.affilter import filter_alphafold_model

app = typer.Typer(no_args_is_help=True)


@app.command()
def filter(
    input_pdb: Path = typer.Argument(..., help="Input AlphaFold PDB file"),
    output_pdb: Path = typer.Option(Path("alphafold_filtered.pdb"), "--output", help="Output filtered PDB file"),
    plddt_threshold: float = typer.Option(0.5, "--plddt-threshold", help="Minimum pLDDT to keep (0-1)"),
    filter_loops: bool = typer.Option(True, "--filter-loops/--no-filter-loops", help="Filter extended loops"),
    filter_connectivity: bool = typer.Option(True, "--filter-connectivity/--no-filter-connectivity", help="Filter low-connectivity regions"),
    max_ca_distance: float = typer.Option(4.5, "--max-ca-distance", help="Max C-alpha distance for loop detection (Å)"),
    min_loop_length: int = typer.Option(10, "--min-loop-length", help="Minimum loop length to filter"),
    max_loop_length: int = typer.Option(50, "--max-loop-length", help="Maximum loop length to filter"),
    connectivity_threshold: float = typer.Option(6.0, "--connectivity-threshold", help="Distance threshold for connectivity (Å)"),
    min_neighbors: int = typer.Option(2, "--min-neighbors", help="Minimum neighbors for connectivity"),
    clustering_method: str = typer.Option("dbscan", "--clustering", help="Clustering method (dbscan|agglomerative)"),
    clustering_eps: float = typer.Option(15.0, "--clustering-eps", help="DBSCAN eps parameter (Å)"),
    clustering_min_samples: int = typer.Option(10, "--clustering-min-samples", help="DBSCAN min_samples parameter"),
    n_clusters: Optional[int] = typer.Option(None, "--n-clusters", help="Number of clusters for agglomerative (None = auto)"),
    out_dir: Path = typer.Option(Path("affilter_outputs"), "--out-dir", help="Output directory"),
):
    """Filter AlphaFold model and identify domains.
    
    Removes low pLDDT regions, extended loops, and low-connectivity artifacts.
    Identifies domains using clustering analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Filtering AlphaFold model: {input_pdb}")
    typer.echo(f"  pLDDT threshold: {plddt_threshold}")
    typer.echo(f"  Filter extended loops: {filter_loops}")
    typer.echo(f"  Filter low connectivity: {filter_connectivity}")
    
    # Filter model
    filtered_model = filter_alphafold_model(
        pdb_path=input_pdb,
        plddt_threshold=plddt_threshold,
        filter_extended_loops=filter_loops,
        filter_low_connectivity=filter_connectivity,
        max_ca_distance=max_ca_distance,
        min_loop_length=min_loop_length,
        max_loop_length=max_loop_length,
        connectivity_threshold=connectivity_threshold,
        min_neighbors=min_neighbors,
        clustering_method=clustering_method,
        clustering_eps=clustering_eps,
        clustering_min_samples=clustering_min_samples,
        n_clusters=n_clusters,
    )
    
    # Write filtered PDB
    output_path = out_dir / output_pdb.name if output_pdb.parent == Path(".") else output_pdb
    filtered_model.structure.write_pdb(str(output_path))
    typer.echo(f"\nFiltered model written to: {output_path}")
    
    # Write statistics
    stats_path = out_dir / "affilter_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"AlphaFold Model Filtering Statistics\n")
        f.write(f"=====================================\n\n")
        f.write(f"Input file: {input_pdb}\n")
        f.write(f"Output file: {output_path}\n\n")
        f.write(f"Filtering parameters:\n")
        f.write(f"  pLDDT threshold: {plddt_threshold}\n")
        f.write(f"  Filter extended loops: {filter_loops}\n")
        f.write(f"  Filter low connectivity: {filter_connectivity}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Filtered residues: {len(filtered_model.filtered_residues)}\n")
        f.write(f"  Removed residues: {len(filtered_model.removed_residues)}\n")
        f.write(f"  Low pLDDT regions: {len(filtered_model.low_plddt_regions)}\n")
        f.write(f"  Identified domains: {len(filtered_model.domains)}\n\n")
        
        if filtered_model.low_plddt_regions:
            f.write(f"Low pLDDT regions (for loop modeling):\n")
            for chain_id, start_resi, end_resi in filtered_model.low_plddt_regions:
                f.write(f"  {chain_id}:{start_resi}-{end_resi}\n")
            f.write("\n")
        
        if filtered_model.domains:
            f.write(f"Identified domains:\n")
            for domain in filtered_model.domains:
                f.write(f"  Domain {domain.domain_id} ({domain.chain_id}:{domain.start_resi}-{domain.end_resi})\n")
                f.write(f"    Residues: {len(domain.residue_indices)}\n")
                f.write(f"    Radius of gyration: {domain.radius_of_gyration:.2f} Å\n")
                f.write(f"    pLDDT: {domain.plddt_min:.2f}-{domain.plddt_max:.2f} (mean: {domain.plddt_mean:.2f})\n")
                f.write(f"    Centroid: ({domain.centroid[0]:.2f}, {domain.centroid[1]:.2f}, {domain.centroid[2]:.2f})\n\n")
    
    typer.echo(f"Statistics written to: {stats_path}")
    
    # Write domains CSV
    if filtered_model.domains:
        domains_csv = out_dir / "affilter_domains.csv"
        rows = []
        for domain in filtered_model.domains:
            rows.append({
                'domain_id': domain.domain_id,
                'chain_id': domain.chain_id,
                'start_resi': domain.start_resi,
                'end_resi': domain.end_resi,
                'n_residues': len(domain.residue_indices),
                'radius_of_gyration': domain.radius_of_gyration,
                'plddt_mean': domain.plddt_mean,
                'plddt_min': domain.plddt_min,
                'plddt_max': domain.plddt_max,
                'centroid_x': domain.centroid[0],
                'centroid_y': domain.centroid[1],
                'centroid_z': domain.centroid[2],
            })
        df = pd.DataFrame(rows)
        df.to_csv(domains_csv, index=False)
        typer.echo(f"Domains CSV written to: {domains_csv}")
    
    # Write low pLDDT regions CSV (for loop modeling integration)
    if filtered_model.low_plddt_regions:
        low_plddt_csv = out_dir / "affilter_low_plddt_regions.csv"
        rows = []
        for chain_id, start_resi, end_resi in filtered_model.low_plddt_regions:
            rows.append({
                'chain_id': chain_id,
                'start_resi': start_resi,
                'end_resi': end_resi,
                'length': end_resi - start_resi + 1,
            })
        df = pd.DataFrame(rows)
        df.to_csv(low_plddt_csv, index=False)
        typer.echo(f"Low pLDDT regions CSV written to: {low_plddt_csv}")
        typer.echo(f"  (Use these regions with loopcloud to rebuild missing segments)")
    
    typer.echo(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    app()

