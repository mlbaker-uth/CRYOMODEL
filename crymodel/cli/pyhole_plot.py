# crymodel/cli/pyhole_plot.py
"""CLI command for plotting pyHole results."""
from __future__ import annotations
from pathlib import Path
import typer
import matplotlib.pyplot as plt

from ..pore.plotter import (
    _load_one,
    _parse_pair_any,
    _parse_hlines,
    _apply_paper_style,
    plot_single,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def plot(
    inputs: str = typer.Argument(..., help="Comma-separated list of pyHole output prefixes or CSV/JSON files"),
    out: str = typer.Option("pyhole_plot", "--out", help="Output basename (without extension)"),
    overlay: bool = typer.Option(False, "--overlay", help="Overlay all inputs in one axes"),
    grid: str = typer.Option(None, "--grid", help="Grid like '1x5' for multi-panel layout"),
    labels: str = typer.Option(None, "--labels", help="Comma-separated labels matching inputs"),
    titles: str = typer.Option(None, "--titles", help="Comma-separated titles (grid mode)"),
    ylim: str = typer.Option("0.5,8.0", "--ylim", help="Radius range (lo,hi)"),
    species: str = typer.Option("water", "--species", help="Passability species to shade"),
    hlines: str = typer.Option("1.4:water", "--hlines", help="Reference lines 'y[:label],y[:label],...'"),
    pdf: bool = typer.Option(False, "--pdf", help="Also save PDF alongside PNG"),
    style_paper: bool = typer.Option(False, "--style-paper", help="Apply compact journal-like styling"),
    swap_axes: bool = typer.Option(False, "--swap-axes", help="Swap axes: radius on X, s_A on Y"),
    secondary: str = typer.Option(None, "--secondary", help="Secondary curve: 'hydro', 'electro', or 'occ'"),
    sec_ylim: str = typer.Option(None, "--sec-ylim", help="Limits for secondary axis (lo,hi)"),
    sec_label: str = typer.Option(None, "--sec-label", help="Override label for secondary axis"),
    sec_order: str = typer.Option("asc", "--sec-order", help="Secondary axis direction: 'asc' or 'desc'"),
    primary_color: str = typer.Option("black", "--primary-color", help="Primary curve color(s), comma-separated for multiple"),
    secondary_color: str = typer.Option(None, "--secondary-color", help="Secondary curve color"),
):
    """Plot pyHole radius profiles from CSV/summary outputs."""
    if style_paper:
        _apply_paper_style()
    
    # Parse inputs
    input_list = [s.strip() for s in inputs.split(',')]
    
    # Parse limits
    ylim_parsed = _parse_pair_any(ylim, flag_name='ylim')
    sec_ylim_parsed = _parse_pair_any(sec_ylim, flag_name='sec_ylim') if sec_ylim else None
    
    labels_list = [s.strip() for s in labels.split(',')] if labels else None
    titles_list = [s.strip() for s in titles.split(',')] if titles else None
    hlines_parsed = _parse_hlines(hlines)
    primary_colors = [c.strip() for c in primary_color.split(',')] if primary_color else ['black']
    
    # Load datasets
    datasets = []
    for inp in input_list:
        df, summary, label = _load_one(inp)
        datasets.append((df, summary, label))
    
    # Overlay mode
    if overlay:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, (df, summary, label) in enumerate(datasets):
            lbl = labels_list[i] if labels_list and i < len(labels_list) else label
            color = primary_colors[i % len(primary_colors)] if primary_colors else 'black'
            plot_single(
                df, summary, ax=ax, title=None, ylim=ylim_parsed,
                species=species if species else None,
                hlines=hlines_parsed, color=color, label=lbl,
                swap_axes=swap_axes, secondary=secondary,
                sec_ylim=sec_ylim_parsed, sec_label=sec_label,
                sec_color=secondary_color, sec_order=sec_order
            )
        fig.tight_layout()
        out_png = f"{out}.png"
        plt.savefig(out_png)
        if pdf:
            plt.savefig(f"{out}.pdf")
        typer.echo(f"Wrote {out_png}")
        return
    
    # Grid mode
    if grid:
        try:
            nr, nc = grid.lower().split('x')
            nr, nc = int(nr), int(nc)
        except Exception:
            typer.echo("ERROR: --grid must be like '1x5' or '2x3'", err=True)
            raise typer.Exit(1)
        if nr * nc < len(datasets):
            typer.echo(f"ERROR: --grid {nr}x{nc} has fewer panels than inputs ({len(datasets)})", err=True)
            raise typer.Exit(1)
        fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.6, nr * 2.8), squeeze=False)
        for idx, (df, summary, label) in enumerate(datasets):
            r, c = divmod(idx, nc)
            ttl = titles_list[idx] if titles_list and idx < len(titles_list) else label
            color = primary_colors[idx % len(primary_colors)] if primary_colors else 'black'
            plot_single(
                df, summary, ax=axes[r][c], title=ttl, ylim=ylim_parsed,
                species=species if species else None,
                hlines=hlines_parsed, color=color, label=None,
                swap_axes=swap_axes, secondary=secondary,
                sec_ylim=sec_ylim_parsed, sec_label=sec_label,
                sec_color=secondary_color, sec_order=sec_order
            )
        # Hide unused axes
        for idx in range(len(datasets), nr * nc):
            r, c = divmod(idx, nc)
            axes[r][c].axis('off')
        fig.tight_layout()
        out_png = f"{out}.png"
        plt.savefig(out_png)
        if pdf:
            plt.savefig(f"{out}.pdf")
        typer.echo(f"Wrote {out_png}")
        return
    
    # Single plot
    df, summary, label = datasets[0]
    color = primary_colors[0] if primary_colors else 'black'
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    plot_single(
        df, summary, ax=ax, title=label, ylim=ylim_parsed,
        species=species if species else None,
        hlines=hlines_parsed, color=color, label=None,
        swap_axes=swap_axes, secondary=secondary,
        sec_ylim=sec_ylim_parsed, sec_label=sec_label,
        sec_color=secondary_color, sec_order=sec_order
    )
    fig.tight_layout()
    out_png = f"{out}.png"
    plt.savefig(out_png)
    if pdf:
        plt.savefig(f"{out}.pdf")
    typer.echo(f"Wrote {out_png}")


if __name__ == "__main__":
    app()

