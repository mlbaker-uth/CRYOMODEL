# crymodel/cli/train_ensemble.py
"""CLI command for training ensemble of models."""
from __future__ import annotations
import typer

from ..ml.ensemble import train_ensemble

app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    train_csv: str = typer.Option(..., "--train-csv", help="Features CSV with labels"),
    outdir: str = typer.Option("ensemble_model", "--outdir", help="Output directory for ensemble"),
    n_models: int = typer.Option(3, "--n-models", help="Number of models in ensemble"),
    epochs: int = typer.Option(50, "--epochs", help="Number of training epochs per model"),
    batch: int = typer.Option(512, "--batch", help="Batch size"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    focal: bool = typer.Option(True, "--focal/--no-focal", help="Use focal loss"),
    class_weights: bool = typer.Option(True, "--class-weights/--no-class-weights", help="Use class weights"),
    group_col: str = typer.Option("pdb_id", "--group-col", help="Column for group-based splitting"),
):
    """Train an ensemble of models for improved predictions."""
    train_ensemble(
        train_csv=train_csv,
        outdir=outdir,
        n_models=n_models,
        epochs=epochs,
        batch=batch,
        lr=lr,
        focal=focal,
        class_weights=class_weights,
        group_col=group_col,
    )


if __name__ == "__main__":
    app()

