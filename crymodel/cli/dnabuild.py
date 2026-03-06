from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import typer

from ..nucleotide.dna_builder import build_poly_at_dna

app = typer.Typer(no_args_is_help=True)

DEFAULT_TEMPLATE_PDB = Path(__file__).resolve().parents[2] / "data" / "DNA-TEMPLATES" / "1BNA.pdb"


def _read_sequences(seq_file: Path) -> Tuple[Optional[str], Optional[str]]:
    lines = [l.strip() for l in seq_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if len(lines) < 2:
        raise typer.BadParameter("Sequence file must contain two sequences (one per strand).")
    return lines[0].upper(), lines[1].upper()


@app.command()
def build(
    map_path: str = typer.Option(..., "--map", help="Input map (.mrc) containing dsDNA only"),
    n_basepairs: int = typer.Option(..., "--n-bp", help="Number of base pairs to build"),
    threshold: float = typer.Option(..., "--threshold", help="Density threshold for peak detection"),
    out_pdb: str = typer.Option("dna_model.pdb", "--out-pdb", help="Output PDB path"),
    sequence_file: Optional[str] = typer.Option(None, "--sequence-file", help="Optional file with two strand sequences"),
    template_pdb: str = typer.Option(
        str(DEFAULT_TEMPLATE_PDB),
        "--template-pdb",
        help="Template PDB containing dsDNA (used to extract base pair)",
    ),
    min_distance_vox: int = typer.Option(4, "--min-distance-vox", help="Minimum distance between base pairs (voxels)"),
    resolution: float = typer.Option(3.0, "--resolution", help="Resolution for template density (Å)"),
    output_swapped: bool = typer.Option(
        False,
        "--output-swapped/--no-output-swapped",
        help="Output swapped-strand model when sequence file provided",
    ),
):
    map_path = Path(map_path)
    if not map_path.exists():
        raise typer.BadParameter(f"Map not found: {map_path}")

    template_pdb = Path(template_pdb)
    if not template_pdb.exists():
        raise typer.BadParameter(f"Template PDB not found: {template_pdb}")

    seq_a = seq_b = None
    if sequence_file:
        seq_path = Path(sequence_file)
        if not seq_path.exists():
            raise typer.BadParameter(f"Sequence file not found: {seq_path}")
        seq_a, seq_b = _read_sequences(seq_path)

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    outputs = build_poly_at_dna(
        map_path=map_path,
        n_basepairs=n_basepairs,
        threshold=threshold,
        out_pdb=out_pdb,
        template_pdb=template_pdb,
        sequence_a=seq_a,
        sequence_b=seq_b,
        min_distance_vox=min_distance_vox,
        resolution_A=resolution,
        output_swapped=output_swapped and sequence_file is not None,
    )

    for out in outputs:
        typer.echo(f"Wrote: {out}")

