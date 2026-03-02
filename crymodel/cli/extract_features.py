# crymodel/cli/extract_features.py
"""CLI command for extracting features from training PDB files."""
from __future__ import annotations
import typer
from pathlib import Path
import pandas as pd

from ..ml.extract_features import extract_features_batch

app = typer.Typer(no_args_is_help=True)


@app.command()
def extract(
    pdb_dir: str = typer.Option(..., "--pdb-dir", help="Directory containing PDB files"),
    output_csv: str = typer.Option(..., "--output-csv", help="Output CSV path"),
    resolution_csv: str = typer.Option(None, "--resolution-csv", help="Optional CSV with pdb_id and resolution columns"),
    pdb_ids_file: str = typer.Option(None, "--pdb-ids-file", help="Optional file with PDB IDs to process (one per line)"),
    max_structures: int = typer.Option(None, "--max-structures", help="Limit number of structures to process"),
    keep_hydrogens: bool = typer.Option(False, "--keep-hydrogens", help="Keep hydrogen atoms (default: remove them)"),
):
    """Extract features from PDB files for training."""
    pdb_dir_path = Path(pdb_dir)
    output_csv_path = Path(output_csv)
    resolution_csv_path = Path(resolution_csv) if resolution_csv else None
    
    # Get PDB IDs if provided
    pdb_ids = None
    if pdb_ids_file:
        pdb_ids_file_path = Path(pdb_ids_file)
        if pdb_ids_file_path.exists():
            pdb_ids = [line.strip().upper() for line in pdb_ids_file_path.open() if line.strip()]
    
    extract_features_batch(
        pdb_dir=pdb_dir_path,
        output_csv=output_csv_path,
        pdb_ids=pdb_ids,
        resolution_csv=resolution_csv_path,
        remove_hydrogens=not keep_hydrogens,
        max_structures=max_structures,
    )


if __name__ == "__main__":
    app()

