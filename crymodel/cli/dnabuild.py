from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import typer

from ..nucleotide.dna_builder import build_poly_at_dna, build_poly_at_from_2bp_centerline
from .command_log import log_command

app = typer.Typer(no_args_is_help=True)

DEFAULT_TEMPLATE_PDB = Path(__file__).resolve().parents[2] / "data" / "DNA-TEMPLATES" / "AT-template.pdb"
DEFAULT_TEMPLATE_2BP_PDB = Path(__file__).resolve().parents[2] / "data" / "DNA-TEMPLATES" / "2AT-template.pdb"


def _read_sequences(seq_file: Path) -> Tuple[Optional[str], Optional[str]]:
    lines = [l.strip() for l in seq_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if len(lines) < 2:
        raise typer.BadParameter("Sequence file must contain two sequences (one per strand).")
    return lines[0].upper(), lines[1].upper()


@app.command()
@log_command("dnabuild build")
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


@app.command("build-2bp")
@log_command("dnabuild build-2bp")
def build_2bp(
    centerline_pdb: str = typer.Option(..., "--centerline-pdb", help="Centerline PDB (polyline)"),
    template_2bp_pdb: str = typer.Option(
        str(DEFAULT_TEMPLATE_2BP_PDB),
        "--template-2bp-pdb",
        help="2-bp template PDB (default: data/DNA-TEMPLATES/2AT-template.pdb)",
    ),
    out_pdb: str = typer.Option("polyAT_2bp.pdb", "--out-pdb", help="Output PDB path"),
    report: Optional[str] = typer.Option(None, "--report", help="Optional report output path"),
    map_path: Optional[str] = typer.Option(None, "--map", help="Optional map for phase optimization"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Threshold context for report"),
    n_bp: Optional[int] = typer.Option(None, "--n-bp", help="Number of base pairs (optional)"),
    target_spacing: float = typer.Option(3.4, "--target-spacing", help="Target spacing between base pairs (Å)"),
    twist_deg: float = typer.Option(35.0, "--twist-deg", help="Twist per base pair (degrees)"),
    trim_start_bp: int = typer.Option(0, "--trim-start-bp", help="Trim start by base pairs"),
    trim_end_bp: int = typer.Option(0, "--trim-end-bp", help="Trim end by base pairs"),
    trim_start_A: float = typer.Option(0.0, "--trim-start-A", help="Trim start by Å"),
    trim_end_A: float = typer.Option(0.0, "--trim-end-A", help="Trim end by Å"),
    global_phase_step_deg: float = typer.Option(5.0, "--global-phase-step-deg", help="Global phase step (deg)"),
    no_global_phase_opt: bool = typer.Option(False, "--no-global-phase-opt", help="Disable global phase optimization"),
    local_refine: bool = typer.Option(False, "--local-refine", help="Enable local refine"),
    local_shift_A: float = typer.Option(1.0, "--local-shift-A", help="Local shift max (Å)"),
    local_shift_step_A: float = typer.Option(0.5, "--local-shift-step-A", help="Local shift step (Å)"),
    local_twist_deg: float = typer.Option(10.0, "--local-twist-deg", help="Local twist max (deg)"),
    local_twist_step_deg: float = typer.Option(2.5, "--local-twist-step-deg", help="Local twist step (deg)"),
    backbone_only_score: bool = typer.Option(False, "--backbone-only-score", help="Score using backbone only"),
):
    centerline_pdb = Path(centerline_pdb)
    if not centerline_pdb.exists():
        raise typer.BadParameter(f"Centerline PDB not found: {centerline_pdb}")

    template_2bp_pdb = Path(template_2bp_pdb)
    if not template_2bp_pdb.exists():
        raise typer.BadParameter(f"Template PDB not found: {template_2bp_pdb}")

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    map_path_obj = Path(map_path) if map_path else None
    if map_path_obj and not map_path_obj.exists():
        raise typer.BadParameter(f"Map not found: {map_path_obj}")

    report_path = Path(report) if report else None

    out_path, report_out = build_poly_at_from_2bp_centerline(
        centerline_pdb=centerline_pdb,
        template_2bp_pdb=template_2bp_pdb,
        out_pdb=out_pdb,
        report_path=report_path,
        map_path=map_path_obj,
        threshold=threshold,
        n_bp=n_bp,
        target_spacing=target_spacing,
        twist_deg=twist_deg,
        trim_start_bp=trim_start_bp,
        trim_end_bp=trim_end_bp,
        trim_start_A=trim_start_A,
        trim_end_A=trim_end_A,
        global_phase_step_deg=global_phase_step_deg,
        no_global_phase_opt=no_global_phase_opt,
        local_refine=local_refine,
        local_shift_A=local_shift_A,
        local_shift_step_A=local_shift_step_A,
        local_twist_deg=local_twist_deg,
        local_twist_step_deg=local_twist_step_deg,
        backbone_only_score=backbone_only_score,
    )

    typer.echo(f"Wrote: {out_path}")
    if report_out:
        typer.echo(f"Wrote: {report_out}")

