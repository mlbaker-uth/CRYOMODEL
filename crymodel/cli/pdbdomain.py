"""CLI for automatic PDB domain identification."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
import gemmi

from ..domains.domain_identifier import (
    identify_domains,
    parse_sse_from_pdb_header,
    parse_sse_with_dssp,
    write_domain_spec,
    write_domain_csv,
    write_domain_pdb,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def identify(
    model: str = typer.Option(..., "--model", help="Input model PDB/mmCIF"),
    chain: Optional[str] = typer.Option(None, "--chain", help="Chain ID (default: first chain)"),
    n_domains: Optional[int] = typer.Option(None, "--n-domains", help="Force number of domains"),
    merge_distance: float = typer.Option(25.0, "--merge-distance", help="Merge distance (Å) for domain clustering"),
    seed_size: int = typer.Option(20, "--seed-size", help="Residues per seed segment"),
    min_domain_residues: int = typer.Option(50, "--min-domain-residues", help="Merge smaller domains into nearest"),
    prefer_gaps: bool = typer.Option(True, "--prefer-gaps/--no-prefer-gaps", help="Prefer domain breaks at sequence gaps"),
    gap_window: int = typer.Option(10, "--gap-window", help="Residues to search for nearest gap"),
    gaps_only: bool = typer.Option(False, "--gaps-only/--no-gaps-only", help="Only allow domain breaks at sequence gaps"),
    sse_source: str = typer.Option("header", "--sse-source", help="SSE source: header, dssp, auto, or none"),
    sse_window: int = typer.Option(10, "--sse-window", help="Residues to search for non-SSE boundary"),
    out_prefix: str = typer.Option("domains", "--out-prefix", help="Output file prefix"),
    write_pdb: bool = typer.Option(True, "--write-pdb/--no-write-pdb", help="Write PDB with domain IDs in B-factor"),
):
    """Identify structural domains from a model and write domain specification."""
    model_path = Path(model)
    if not model_path.exists():
        raise typer.BadParameter(f"Model not found: {model_path}")

    structure = gemmi.read_structure(str(model_path))
    if chain is None:
        chain = structure[0][0].name if len(structure[0]) > 0 else None
    if chain is None:
        raise typer.BadParameter("No chains found in model.")

    sse_resnums = None
    source = sse_source.lower()
    if source == "header" and model_path.suffix.lower() == ".pdb":
        sse_resnums = parse_sse_from_pdb_header(model_path, chain)
    elif source == "dssp":
        sse_resnums = parse_sse_with_dssp(model_path, chain)
    elif source == "auto":
        if model_path.suffix.lower() == ".pdb":
            sse_resnums = parse_sse_from_pdb_header(model_path, chain)
        if not sse_resnums:
            sse_resnums = parse_sse_with_dssp(model_path, chain)

    records, ranges_by_domain = identify_domains(
        structure=structure,
        chain_id=chain,
        seed_size=seed_size,
        n_domains=n_domains,
        merge_distance=merge_distance,
        min_domain_residues=min_domain_residues,
        prefer_gaps=prefer_gaps,
        gap_window=gap_window,
        gaps_only=gaps_only,
        sse_resnums=sse_resnums,
        sse_window=sse_window,
    )

    out_base = Path(out_prefix)
    json_path = out_base.with_suffix(".json")
    csv_path = out_base.with_suffix(".csv")
    pdb_path = out_base.with_suffix(".pdb")

    write_domain_spec(json_path, chain, ranges_by_domain)
    write_domain_csv(csv_path, records, ranges_by_domain)
    if write_pdb:
        write_domain_pdb(structure, chain, ranges_by_domain, pdb_path)

    typer.echo(f"Wrote: {json_path}")
    typer.echo(f"Wrote: {csv_path}")
    if write_pdb:
        typer.echo(f"Wrote: {pdb_path}")


if __name__ == "__main__":
    app()
