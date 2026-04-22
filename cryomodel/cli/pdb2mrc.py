"""CLI command for converting model coordinates to a synthetic density map."""
from __future__ import annotations

from pathlib import Path
import math
import typer
import numpy as np
import gemmi
from scipy.ndimage import gaussian_filter

from cryomodel.io.mrc import MapVolume, write_map

app = typer.Typer(no_args_is_help=True)


def _collect_atoms(structure: gemmi.Structure) -> list[tuple[np.ndarray, float, float]]:
    atoms: list[tuple[np.ndarray, float, float]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = np.array([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)], dtype=np.float32)
                    occ = float(getattr(atom, "occ", 1.0) or 1.0)
                    b_iso = float(getattr(atom, "b_iso", 0.0) or 0.0)
                    atoms.append((pos, occ, b_iso))
    return atoms


def _trilinear_add(grid_zyx: np.ndarray, xyz_vox: np.ndarray, value: float) -> None:
    """Deposit into a C-order (z,y,x) volume: x varies fastest in MRC2000/CCP4 layout."""
    x, y, z = float(xyz_vox[0]), float(xyz_vox[1]), float(xyz_vox[2])
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    z0 = int(math.floor(z))
    dx = x - x0
    dy = y - y0
    dz = z - z0

    nz, ny, nx = grid_zyx.shape
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                ix = x0 + ox
                iy = y0 + oy
                iz = z0 + oz
                if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                    continue
                wx = (1.0 - dx) if ox == 0 else dx
                wy = (1.0 - dy) if oy == 0 else dy
                wz = (1.0 - dz) if oz == 0 else dz
                grid_zyx[iz, iy, ix] += float(value) * wx * wy * wz


@app.command()
def convert(
    model: Path = typer.Option(..., "--model", help="Input model PDB/mmCIF"),
    output_map: Path = typer.Option(Path("model_density.mrc"), "--output-map", help="Output map (.mrc)"),
    resolution: float = typer.Option(3.0, "--resolution", help="Target map resolution (A, FWHM)"),
    apix: float = typer.Option(1.0, "--apix", help="Sampling (A/voxel)"),
    box: int = typer.Option(0, "--box", help="Cubic box size in voxels (0 = auto)"),
    center: bool = typer.Option(False, "--center/--no-center", help="Recenter model into box (default: preserve coordinate frame)"),
    scale_occupancy: bool = typer.Option(False, "--scale-occupancy/--no-scale-occupancy", help="Scale atom weights by occupancy"),
    scale_bfactor: bool = typer.Option(False, "--scale-bfactor/--no-scale-bfactor", help="Scale atom weights by isotropic B-factor"),
    origin_mode: str = typer.Option("auto", "--origin-mode", help="Origin convention: auto|half-box-shift|zero"),
    normalize_max: bool = typer.Option(True, "--normalize-max/--no-normalize-max", help="Normalize output map max to 1.0"),
) -> None:
    """Convert a model to synthetic density map (even, cubic MRC output)."""
    model = Path(model).expanduser()
    if not model.exists():
        typer.echo(f"ERROR: Model not found: {model}", err=True)
        raise typer.Exit(1)
    if resolution <= 0 or apix <= 0:
        typer.echo("ERROR: resolution and apix must be > 0", err=True)
        raise typer.Exit(1)

    structure = gemmi.read_structure(str(model))
    atoms = _collect_atoms(structure)
    if not atoms:
        typer.echo("ERROR: No atoms found in model.", err=True)
        raise typer.Exit(1)

    coords = np.stack([a[0] for a in atoms], axis=0)
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    xyz_center = 0.5 * (xyz_min + xyz_max)
    extent = xyz_max - xyz_min

    if box and box > 0:
        n = int(box)
    else:
        # Padding by ~2 * resolution around model envelope.
        side_a = float(np.max(extent) + 2.0 * resolution)
        n = int(math.ceil(side_a / apix))
    if n < 8:
        n = 8
    # Enforce cubic even box.
    if n % 2 == 1:
        n += 1

    # MRC2000 / EMAN2 / cryomodel: numpy volume [nz, ny, nx] (z slowest, x fastest in flat order).
    grid_zyx = np.zeros((n, n, n), dtype=np.float32)
    box_center_xyz = np.array([0.5 * n * apix, 0.5 * n * apix, 0.5 * n * apix], dtype=np.float32)
    if center:
        origin_xyz = xyz_center - box_center_xyz
    else:
        origin_xyz = xyz_min - np.array([resolution, resolution, resolution], dtype=np.float32)

    # Deposit atoms as weighted impulses with trilinear assignment.
    # Then blur globally to target resolution.
    bfactor_scale_const = 1.0 / (4.0 * float(resolution) * float(resolution))
    for pos, occ, b_iso in atoms:
        weight = 1.0
        if scale_occupancy:
            weight *= occ if occ > 0 else 1.0
        if scale_bfactor:
            weight *= math.exp(-max(0.0, float(b_iso)) * bfactor_scale_const)
        vox = (pos - origin_xyz) / float(apix)
        _trilinear_add(grid_zyx, vox, weight)

    sigma_vox = float(resolution) / (2.355 * float(apix))
    if sigma_vox > 0:
        grid_zyx = gaussian_filter(grid_zyx, sigma=sigma_vox, mode="constant", cval=0.0).astype(np.float32)

    if normalize_max:
        vmax = float(np.max(grid_zyx))
        if vmax > 0:
            grid_zyx /= vmax

    out_origin = origin_xyz.astype(np.float32).copy()
    origin_mode_norm = (origin_mode or "auto").strip().lower()
    if origin_mode_norm == "half-box-shift":
        # Compatibility mode for viewers/conventions that effectively expect an additional
        # half-box negative shift relative to model-frame origin.
        out_origin = out_origin - np.array([0.5 * n * apix, 0.5 * n * apix, 0.5 * n * apix], dtype=np.float32)
    elif origin_mode_norm == "zero":
        out_origin = np.zeros(3, dtype=np.float32)
    elif origin_mode_norm != "auto":
        typer.echo(f"ERROR: Unknown --origin-mode {origin_mode!r}. Use auto|half-box-shift|zero", err=True)
        raise typer.Exit(1)

    output_map = Path(output_map).expanduser()
    output_map.parent.mkdir(parents=True, exist_ok=True)

    out_origin_f = out_origin.astype(np.float32)
    mv = MapVolume(data_zyx=grid_zyx, apix=float(apix), origin_xyzA=out_origin_f)
    write_map(output_map, mv, grid_zyx)

    typer.echo(f"Wrote: {output_map.resolve()}")
    typer.echo(f"Box: {n} x {n} x {n} voxels (even, cubic)")
    typer.echo(f"apix: {apix:.4f} A/voxel, resolution: {resolution:.3f} A")
    typer.echo(f"origin mode: {origin_mode_norm}")
    typer.echo(f"origin (x,y,z): ({out_origin[0]:.3f}, {out_origin[1]:.3f}, {out_origin[2]:.3f})")


if __name__ == "__main__":
    app()
