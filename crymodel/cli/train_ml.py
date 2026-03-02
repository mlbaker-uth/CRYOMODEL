# crymodel/cli/train_ml.py
"""CLI command for training ion/water classification model."""
from __future__ import annotations
import typer
from pathlib import Path

from ..ml.train import main as train_main

app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    train_csv: str = typer.Option(..., "--train-csv", help="Features CSV with labels"),
    outdir: str = typer.Option("ionwater_env_model", "--outdir", help="Output directory for model"),
    epochs: int = typer.Option(40, "--epochs", help="Number of training epochs"),
    batch: int = typer.Option(512, "--batch", help="Batch size"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    focal: bool = typer.Option(False, "--focal", help="Use focal loss"),
    class_weights: bool = typer.Option(False, "--class-weights", help="Use class weights"),
    group_col: str = typer.Option("pdb_id", "--group-col", help="Column for group-based splitting"),
):
    """Train ion/water classification model."""
    train_main(
        train_csv=train_csv,
        outdir=outdir,
        epochs=epochs,
        batch=batch,
        lr=lr,
        focal=focal,
        class_weights=class_weights,
        group_col=group_col,
    )


if __name__ == "__main__":
    app()

