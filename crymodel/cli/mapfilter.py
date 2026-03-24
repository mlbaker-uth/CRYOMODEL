# crymodel/cli/mapfilter.py
"""CLI for generic map filtering: lowpass, highpass, Gaussian, threshold, binary, bilateral, Laplacian, etc."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ..io.mrc import read_map, write_map
from ..maps import filters

app = typer.Typer(no_args_is_help=True, help="Apply common cryo-EM map filters.")

FILTER_LIST = ", ".join(filters.FILTER_REGISTRY.keys())


@app.command("apply")
def apply(
    input_map: Path = typer.Argument(..., help="Input MRC/CCP4 map path"),
    output_map: Path = typer.Argument(..., help="Output filtered map path"),
    filter_type: str = typer.Option(..., "--filter", "-f", help=f"Filter type: {FILTER_LIST}"),
    # Resolution-based (Å)
    resolution: Optional[float] = typer.Option(None, "--resolution", "-r", help="Resolution in Å (lowpass/highpass/butterworth)"),
    low_res: Optional[float] = typer.Option(None, "--low-res", help="Bandpass low-resolution cutoff (Å)"),
    high_res: Optional[float] = typer.Option(None, "--high-res", help="Bandpass high-resolution cutoff (Å)"),
    butterworth_order: Optional[int] = typer.Option(None, "--butterworth-order", help="Butterworth order (bandpass/butterworth-*)"),
    gaussian_rolloff: bool = typer.Option(True, "--gaussian-rolloff/--sharp-cutoff", help="Use Gaussian rolloff for lowpass/highpass"),
    # Spatial
    sigma_vox: Optional[float] = typer.Option(None, "--sigma-vox", help="Gaussian/median: sigma or size in voxels"),
    median_size: Optional[int] = typer.Option(None, "--median-size", help="Median filter half-size (cubic 2*size+1)"),
    # Threshold
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Threshold for threshold/binary filters"),
    below_value: float = typer.Option(0.0, "--below-value", help="Value to set below threshold"),
    # Laplacian
    sharpen_strength: float = typer.Option(1.0, "--sharpen-strength", help="Strength for laplacian-sharpen"),
    # Bilateral
    sigma_spatial: Optional[float] = typer.Option(None, "--sigma-spatial", help="Bilateral: spatial sigma (voxels)"),
    sigma_range: Optional[float] = typer.Option(None, "--sigma-range", help="Bilateral: range sigma (intensity)"),
):
    """Apply a filter to a map and write the result.

    Examples:

      crymodel mapfilter apply map.mrc lowpass_4A.mrc -f lowpass -r 4
      crymodel mapfilter apply map.mrc blurred.mrc -f gaussian --sigma-vox 2
      crymodel mapfilter apply map.mrc masked.mrc -f threshold -t 0.5
      crymodel mapfilter apply map.mrc binary.mrc -f binary -t 0.3
      crymodel mapfilter apply map.mrc band.mrc -f bandpass --low-res 10 --high-res 3
    """
    if filter_type not in filters.FILTER_REGISTRY:
        typer.echo(f"Unknown filter '{filter_type}'. Choose from: {FILTER_LIST}", err=True)
        raise typer.Exit(1)

    mv = read_map(input_map)
    data = mv.data_zyx

    # Dispatch with required options per filter
    if filter_type == "lowpass":
        if resolution is None:
            typer.echo("--resolution required for lowpass", err=True)
            raise typer.Exit(1)
        out = filters.lowpass(data, mv.apix, resolution, gaussian=gaussian_rolloff)
    elif filter_type == "highpass":
        if resolution is None:
            typer.echo("--resolution required for highpass", err=True)
            raise typer.Exit(1)
        out = filters.highpass(data, mv.apix, resolution, gaussian=gaussian_rolloff)
    elif filter_type == "bandpass":
        if low_res is None or high_res is None:
            typer.echo("--low-res and --high-res required for bandpass", err=True)
            raise typer.Exit(1)
        out = filters.bandpass(data, mv.apix, low_res, high_res, order=butterworth_order)
    elif filter_type == "gaussian":
        s = sigma_vox if sigma_vox is not None else 1.0
        out = filters.gaussian_filter(data, s)
    elif filter_type == "threshold":
        t = threshold if threshold is not None else 0.0
        out = filters.threshold_filter(data, t, below_value=below_value)
    elif filter_type == "binary":
        if threshold is None:
            typer.echo("--threshold required for binary", err=True)
            raise typer.Exit(1)
        out = filters.binary_filter(data, threshold)
    elif filter_type == "laplacian":
        out = filters.laplacian_filter(data)
    elif filter_type == "laplacian-sharpen":
        out = filters.laplacian_sharpen(data, strength=sharpen_strength)
    elif filter_type == "median":
        size = median_size if median_size is not None else (sigma_vox if sigma_vox is not None else 1)
        out = filters.median_filter(data, int(size))
    elif filter_type == "bilateral":
        sig_s = sigma_spatial if sigma_spatial is not None else 1.0
        out = filters.bilateral_filter_3d(data, sigma_spatial=sig_s, sigma_range=sigma_range)
    elif filter_type == "butterworth-lowpass":
        if resolution is None:
            typer.echo("--resolution required for butterworth-lowpass", err=True)
            raise typer.Exit(1)
        order = butterworth_order or 4
        out = filters.butterworth_lowpass(data, mv.apix, resolution, order=order)
    elif filter_type == "butterworth-highpass":
        if resolution is None:
            typer.echo("--resolution required for butterworth-highpass", err=True)
            raise typer.Exit(1)
        order = butterworth_order or 4
        out = filters.butterworth_highpass(data, mv.apix, resolution, order=order)
    elif filter_type == "normalize":
        out = filters.normalize_zero_mean_unit_variance(data)
    else:
        typer.echo(f"Filter '{filter_type}' not implemented in CLI.", err=True)
        raise typer.Exit(1)

    output_map.parent.mkdir(parents=True, exist_ok=True)
    write_map(output_map, mv, out)
    typer.echo(f"Wrote {output_map} ({filter_type})")


@app.command("list")
def list_filters():
    """List available filter types and their options."""
    typer.echo("Available map filters:\n")
    for name, (_, opts, desc) in filters.FILTER_REGISTRY.items():
        typer.echo(f"  {name}")
        typer.echo(f"    Options: {opts}")
        typer.echo(f"    {desc}\n")
