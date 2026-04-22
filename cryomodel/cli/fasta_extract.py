"""CLI: extract one sequence from a multi-record FASTA (e.g. for pdb-mutate --target-fasta)."""
from __future__ import annotations

from pathlib import Path

import typer

from cryomodel.mutate.align import read_fasta, sequence_from_fasta_row

app = typer.Typer(no_args_is_help=True, help="FASTA utilities")


@app.command("row")
def extract_row(
    input_fasta: Path = typer.Argument(..., exists=True, help="Multi-record FASTA path"),
    out: Path = typer.Argument(..., help="Write a single-record FASTA here"),
    row: int = typer.Option(0, "--row", "-r", min=0, help="0-based record index (file order)"),
) -> None:
    """Write one sequence (gaps stripped, uppercase) as a one-record FASTA."""
    items = list(read_fasta(input_fasta).items())
    if not items:
        raise typer.BadParameter(f"No records in {input_fasta}")
    if row < 0 or row >= len(items):
        raise typer.BadParameter(f"Row {row} out of range (file has {len(items)} record(s))")
    header = items[row][0]
    seq = sequence_from_fasta_row(input_fasta, row)
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f">{header}\n{seq}\n", encoding="utf-8")
    typer.echo(f"Wrote {out} (record {row}: {header})")
