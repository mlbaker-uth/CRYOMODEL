# crymodel/cli/pdbcom.py
"""CLI command for pdbCOM - domain centers of mass."""
from __future__ import annotations
from pathlib import Path
import typer
import gemmi

from ..domains.pdbcom import (
    parse_domain_spec,
    compute_domain_coms,
    write_domain_com_pdb,
    write_domain_com_csv,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def compute(
    model: str = typer.Option(..., "--model", help="Input model PDB/mmCIF"),
    domains: str = typer.Option(..., "--domains", help="Domain specification JSON file"),
    out_prefix: str = typer.Option("domains_com", "--out-prefix", help="Output file prefix"),
    mass_weighted: bool = typer.Option(True, "--mass-weighted/--no-mass-weighted", help="Use mass-weighted COM"),
    atoms: str = typer.Option("all", "--atoms", help="Atom filter: 'all', 'backbone', or 'CA'"),
):
    """Compute domain centers of mass and output as PDB."""
    model_path = Path(model)
    if not model_path.exists():
        typer.echo(f"ERROR: Model not found: {model_path}", err=True)
        raise typer.Exit(1)
    
    domains_path = Path(domains)
    if not domains_path.exists():
        typer.echo(f"ERROR: Domains file not found: {domains_path}", err=True)
        raise typer.Exit(1)
    
    # Load structure
    structure = gemmi.read_structure(str(model_path))
    
    # Parse domains
    domain_spec = parse_domain_spec(domains_path)
    
    # Compute COMs
    typer.echo(f"Computing COMs for {len(domain_spec)} domains...")
    domain_coms = compute_domain_coms(structure, domain_spec, mass_weighted, atoms)
    
    # Write outputs
    out_path = Path(out_prefix)
    pdb_path = out_path.with_suffix('.pdb')
    csv_path = out_path.with_suffix('.csv')
    
    write_domain_com_pdb(domain_coms, pdb_path)
    write_domain_com_csv(domain_coms, csv_path)
    
    typer.echo(f"Wrote: {pdb_path}")
    typer.echo(f"Wrote: {csv_path}")
    
    # Print summary
    typer.echo("\nDomain COMs:")
    for domain_name, data in domain_coms.items():
        x, y, z = data['com']
        typer.echo(f"  {domain_name}: ({x:.2f}, {y:.2f}, {z:.2f}) - {data['num_atoms']} atoms")


if __name__ == "__main__":
    app()

