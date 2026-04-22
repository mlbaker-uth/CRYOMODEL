"""CLI: local χ1 refinement in a spherical zone (cryo-EM map)."""
from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path
from typing import Iterator, Optional, Set

import gemmi
import numpy as np
import typer

from cryomodel.io.mrc import read_map
from cryomodel.validation.ringer_lite import sample_density_at_position
from cryomodel.zonal import (
    parse_center_xyz,
    parse_ncs_chains,
    run_global_zonal_refine,
    run_zonal_chi_refine,
    write_global_result_json,
    write_result_json,
)

app = typer.Typer(no_args_is_help=True, help="Zonal refinement (χ1 in a sphere)")


@app.command("check-map")
def check_map_cmd(
    pdb: Path = typer.Argument(..., exists=True, help="Model (PDB/mmCIF)"),
    map_path: Path = typer.Argument(..., exists=True, help="MRC/CCP4 map"),
    max_ca: int = typer.Option(200, "--max-ca", min=1, max=100000, help="Sample at most this many Cα sites (in order)"),
) -> None:
    """
    Sanity check: mean / median map value at Cα coordinates using the **same** interpolator as
    ``zonal-refine`` (gemmi grid when the map was loaded with ``read_map``).

    If the model is aligned in ChimeraX but mean density here is ~0 everywhere, the frame or file
    may still be wrong—compare with map statistics in the viewer.
    """
    st = gemmi.read_structure(str(pdb))
    if len(st) == 0:
        raise typer.BadParameter("Empty structure.")
    try:
        st.merge_chain_parts()
    except Exception:
        pass
    mv = read_map(map_path)

    def _iter_ca() -> Iterator[gemmi.Atom]:
        for model in st:
            for chain in model:
                for res in chain:
                    try:
                        yield res.sole_atom("CA")
                    except Exception:
                        continue

    vals: list[float] = []
    for ca in islice(_iter_ca(), max_ca):
        p = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float64)
        vals.append(sample_density_at_position(mv, p, density_threshold=0.0))
    if not vals:
        raise typer.BadParameter("No Cα atoms found.")
    arr = np.array(vals, dtype=np.float64)
    mode = "gemmi grid" if mv.grid is not None else "legacy origin+apix"
    typer.echo(
        f"check-map: {len(vals)} Cα  mean={float(arr.mean()):.5g}  median={float(np.median(arr)):.5g}  "
        f"min={float(arr.min()):.5g}  max={float(arr.max()):.5g}  ({mode})"
    )


def _parse_chains(chains: Optional[str]) -> Optional[Set[str]]:
    if not chains or not str(chains).strip():
        return None
    return {c.strip() for c in str(chains).split(",") if c.strip()}


@app.command("run")
def run_cmd(
    pdb: Path = typer.Argument(..., exists=True, help="Input PDB/mmCIF"),
    map_path: Path = typer.Argument(..., exists=True, help="MRC/CCP4 map (same frame as model)"),
    out: Path = typer.Argument(..., help="Output PDB path"),
    center: Optional[str] = typer.Option(
        None,
        "--center",
        "-c",
        help="Sphere center as one comma-separated string x,y,z (Å). Example: --center 113.79,106.93,135.86",
    ),
    cx: Optional[float] = typer.Option(
        None,
        "--cx",
        help="Center x (Å); use with --cy and --cz instead of --center",
    ),
    cy: Optional[float] = typer.Option(
        None,
        "--cy",
        help="Center y (Å)",
    ),
    cz: Optional[float] = typer.Option(
        None,
        "--cz",
        help="Center z (Å)",
    ),
    radius: float = typer.Option(
        ...,
        "--radius",
        "-r",
        min=0.1,
        help="Sphere radius (Å); residues with any atom inside are included",
    ),
    chains: Optional[str] = typer.Option(
        None,
        "--chains",
        help="Comma-separated chain IDs to consider (default: all chains)",
    ),
    passes: int = typer.Option(3, "--passes", min=1, max=50, help="Maximum passes over the zone"),
    weight_map: float = typer.Option(
        0.65,
        "--weight-map",
        help=(
            "Multiplies mean side-chain map value (raw MRC units after --map-density-threshold). "
            "Objective subtracts (weight_map × mean density) vs clash + rotamer; there is no fixed σ scale—"
            "tune to your map range. Values like 5–15 are reasonable if intensities are ~0–1; "
            "use smaller weights if the map is already large (e.g. thousands)."
        ),
    ),
    map_density_threshold: float = typer.Option(
        0.0,
        "--map-density-threshold",
        min=0.0,
        help="Subtract this value from each interpolated density sample (then clamp at 0) before averaging. "
        "Set near background/noise level for continuous cryo-EM maps to limit drift outside real density.",
    ),
    weight_density_anchor: float = typer.Option(
        0.0,
        "--weight-density-anchor",
        min=0.0,
        help=(
            "Penalty weight when mean thresholded map fit drops vs the pose before each χ/rama trial block, "
            "only if that starting mean was above --map-anchor-eps (already in density). "
            "Use with --map-density-threshold; try ~0.5–2.0."
        ),
    ),
    weight_density_gain: float = typer.Option(
        0.0,
        "--weight-density-gain",
        min=0.0,
        help=(
            "Bonus weight for improving mean thresholded map fit when the starting mean was weak (≤ anchor eps); "
            "no penalty for staying weak. Try ~0.3–1.0."
        ),
    ),
    map_anchor_eps: float = typer.Option(
        1e-5,
        "--map-anchor-eps",
        min=0.0,
        help="Divides 'in density' vs 'weak' for map-fit anchoring (same units as thresholded map samples).",
    ),
    weight_rot: float = typer.Option(
        0.15,
        "--weight-rotamer",
        help="Rotamer prior weight (see pdb-mutate χ1 priors)",
    ),
    json_log: Optional[Path] = typer.Option(
        None,
        "--json-log",
        help="Write summary JSON (counts, per-residue score deltas)",
    ),
    soft_buffer: float = typer.Option(
        0.0,
        "--soft-buffer",
        min=0.0,
        help="Extra Å beyond --radius for soft shell (0 = hard zone only, A0).",
    ),
    soft_passes: int = typer.Option(2, "--soft-passes", min=1, max=20, help="Max passes over soft-shell residues"),
    soft_min_clash: float = typer.Option(
        1.0,
        "--soft-min-clash",
        help="Minimum total clash score to consider χ on a soft residue",
    ),
    soft_only_worsened: bool = typer.Option(
        True,
        "--soft-only-worsened/--soft-any-clash",
        help="Only refine soft residues whose clash worsened vs pre-run (baseline); else use absolute threshold only",
    ),
    rama_backbone: bool = typer.Option(
        False,
        "--rama-backbone",
        help="After χ stages, optional small φ/ψ grid to reduce Ramachandran outliers (A2).",
    ),
    rama_step_deg: float = typer.Option(
        3.0,
        "--rama-step-deg",
        min=0.5,
        max=15.0,
        help="Grid step (degrees) for φ/ψ micro-moves.",
    ),
    rama_max_shift_deg: float = typer.Option(
        9.0,
        "--rama-max-shift-deg",
        min=0.0,
        max=30.0,
        help="Maximum |Δφ| and |Δψ| tried (degrees); keep small for near-final models.",
    ),
    weight_rama: float = typer.Option(
        0.08,
        "--weight-rama",
        help="Weight on Ramachandran prior (favored < allowed << outlier).",
    ),
    weight_bb_move: float = typer.Option(
        0.015,
        "--weight-backbone-move",
        help="Penalty on Δφ²+Δψ² (discourages large backbone moves).",
    ),
    rama_include_soft: bool = typer.Option(
        False,
        "--rama-include-soft",
        help="Also run Ramachandran micro-refine on soft-shell residues (default: hard zone only).",
    ),
    rama_nudge_favored: bool = typer.Option(
        False,
        "--rama-nudge-favored",
        help="Allow tiny backbone moves even when already in the favored region.",
    ),
) -> None:
    """
    Refine χ1 side-chain rotamers for residues in a sphere: minimize clash + rotamer prior
    while maximizing mean map value at side-chain atoms (same scoring as pdb-mutate).

    Map sampling uses ``read_map`` + gemmi's grid interpolator (same convention as ChimeraX/gemmi
    for that MRC/CCP4 file), not a hand-rolled origin/apix voxel formula.
    """
    st = gemmi.read_structure(str(pdb))
    if len(st) == 0:
        raise typer.BadParameter("Empty structure.")
    try:
        st.merge_chain_parts()
    except Exception:
        pass
    mv = read_map(map_path)

    have_xyz = all(v is not None for v in (cx, cy, cz))
    have_partial = any(v is not None for v in (cx, cy, cz)) and not have_xyz
    if have_partial:
        raise typer.BadParameter("Provide all three --cx, --cy, and --cz, or use --center x,y,z instead.")
    if have_xyz:
        center_xyz = np.array([float(cx), float(cy), float(cz)], dtype=np.float64)
    elif center is not None and str(center).strip():
        center_xyz = parse_center_xyz(center)
    else:
        raise typer.BadParameter(
            "Sphere center required: use --center x,y,z (comma-separated, no spaces) "
            "or --cx X --cy Y --cz Z (three flags)."
        )

    cf = _parse_chains(chains)

    result = run_zonal_chi_refine(
        st,
        mv,
        center_xyz,
        radius,
        chain_filter=cf,
        passes=passes,
        weight_map=weight_map,
        map_density_threshold=map_density_threshold,
        weight_density_anchor=weight_density_anchor,
        weight_density_gain=weight_density_gain,
        map_anchor_eps=map_anchor_eps,
        weight_rot=weight_rot,
        soft_buffer=soft_buffer,
        soft_passes=soft_passes,
        soft_min_clash=soft_min_clash,
        soft_only_if_worsened=soft_only_worsened,
        rama_backbone=rama_backbone,
        rama_step_deg=rama_step_deg,
        rama_max_shift_deg=rama_max_shift_deg,
        weight_rama=weight_rama,
        weight_bb_move=weight_bb_move,
        rama_include_soft=rama_include_soft,
        rama_nudge_favored=rama_nudge_favored,
    )

    out_p = Path(out).expanduser()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_p))

    typer.echo(
        json.dumps(
            {
                "residues_in_zone": result.residues_in_zone,
                "residues_soft_zone": result.residues_soft_zone,
                "residues_with_chi1": result.residues_with_chi1,
                "residues_soft_with_chi1": result.residues_soft_with_chi1,
                "passes_done": result.passes_done,
                "improvements_last_pass": result.improvements_in_last_pass,
                "soft_passes_done": result.soft_passes_done,
                "improvements_soft_last_pass": result.improvements_soft_last_pass,
                "rama_residues_tried": result.rama_residues_tried,
                "rama_improvements": result.rama_improvements,
                "elapsed_sec": round(result.elapsed_sec, 3),
                "out_pdb": str(out_p),
            },
            indent=2,
        )
    )

    if json_log:
        write_result_json(json_log, result)


@app.command("global")
def global_cmd(
    pdb: Path = typer.Argument(..., exists=True, help="Input PDB/mmCIF (PDB recommended for HELIX/SHEET header zoning)"),
    map_path: Path = typer.Argument(..., exists=True, help="MRC/CCP4 map (same frame as model)"),
    out: Path = typer.Argument(..., help="Output PDB path"),
    ncs: str = typer.Option(
        ...,
        "--ncs",
        help="Master chain first, then NCS copies: e.g. A or A,B,C,D (comma-separated). χ1 is refined on master only, then copied to copies.",
    ),
    target_residues_per_region: int = typer.Option(
        30,
        "--target-residues-per-region",
        min=5,
        max=500,
        help="When --gmm-components is omitted: mixture components K ≈ N_master_Cα / this (indirect zone count).",
    ),
    gmm_components: Optional[int] = typer.Option(
        None,
        "--gmm-components",
        min=1,
        help="Explicit number of GMM components (zones). Overrides the K implied by --target-residues-per-region.",
    ),
    soft_resp_floor: float = typer.Option(
        0.12,
        "--soft-resp-floor",
        min=0.01,
        max=0.99,
        help="Min GMM posterior to include a residue in a region (higher → more overlap).",
    ),
    radius_pad: float = typer.Option(
        4.0,
        "--radius-pad",
        min=0.5,
        max=50.0,
        help="Extra Å added to sphere radius beyond farthest Cα in the region.",
    ),
    max_rounds: int = typer.Option(7, "--max-rounds", min=1, max=100, help="Macro-round cap"),
    converge_rmsd_eps: float = typer.Option(
        0.03,
        "--converge-rmsd-eps",
        min=1e-6,
        max=10.0,
        help="Stop if master-chain Cα RMSD change within a macro-round is below this (Å) for --converge-patience rounds.",
    ),
    converge_patience: int = typer.Option(
        2,
        "--converge-patience",
        min=1,
        max=20,
        help="Consecutive macro-rounds under RMSD threshold before stopping.",
    ),
    random_seed: Optional[int] = typer.Option(
        0,
        "--random-seed",
        help="RNG seed for GMM init and region shuffle (default 0).",
    ),
    sse_header: bool = typer.Option(
        True,
        "--sse-header/--no-sse-header",
        help="Expand GMM regions to full HELIX/SHEET runs from PDB header (master chain only).",
    ),
    gmm_reg_covar: float = typer.Option(
        1e-4,
        "--gmm-reg-covar",
        min=1e-9,
        max=1.0,
        help="sklearn GaussianMixture reg_covar (stability).",
    ),
    ncs_mirror_zones: bool = typer.Option(
        True,
        "--ncs-mirror-zones/--no-ncs-mirror-zones",
        help="With NCS copies: master-only sphere first, then a separate local pass per copy in a sphere built from the same residue patch (recommended for separated homomers).",
    ),
    passes: int = typer.Option(3, "--passes", min=1, max=50, help="χ1 passes per local zone (passed to zonal-refine run)"),
    weight_map: float = typer.Option(0.65, "--weight-map"),
    map_density_threshold: float = typer.Option(0.0, "--map-density-threshold", min=0.0),
    weight_density_anchor: float = typer.Option(0.0, "--weight-density-anchor", min=0.0),
    weight_density_gain: float = typer.Option(0.0, "--weight-density-gain", min=0.0),
    map_anchor_eps: float = typer.Option(1e-5, "--map-anchor-eps", min=0.0),
    weight_rot: float = typer.Option(0.15, "--weight-rotamer"),
    json_log: Optional[Path] = typer.Option(
        None,
        "--json-log",
        help="Write global run summary JSON (rounds, regions, local stats).",
    ),
    soft_buffer: float = typer.Option(0.0, "--soft-buffer", min=0.0),
    soft_passes: int = typer.Option(2, "--soft-passes", min=1, max=20),
    soft_min_clash: float = typer.Option(1.0, "--soft-min-clash"),
    soft_only_worsened: bool = typer.Option(
        True,
        "--soft-only-worsened/--soft-any-clash",
    ),
    rama_backbone: bool = typer.Option(False, "--rama-backbone"),
    rama_step_deg: float = typer.Option(3.0, "--rama-step-deg", min=0.5, max=15.0),
    rama_max_shift_deg: float = typer.Option(9.0, "--rama-max-shift-deg", min=0.0, max=30.0),
    weight_rama: float = typer.Option(0.08, "--weight-rama"),
    weight_bb_move: float = typer.Option(0.015, "--weight-backbone-move"),
    rama_include_soft: bool = typer.Option(False, "--rama-include-soft"),
    rama_nudge_favored: bool = typer.Option(False, "--rama-nudge-favored"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress messages (stderr)."),
) -> None:
    """
    **Global** zonal refinement: fit overlapping 3D GMM zones on the **master** chain Cα cloud,
    shuffle and run local ``zonal-refine`` on each zone, optionally propagate χ1 to NCS copies.

    **Parity with ``zonal-refine run``:** local solver flags match ``run``. With ``--ncs-mirror-zones``
    (default on when copies exist), the **joint** pass is **master-only** in the GMM sphere; each **copy**
    then gets its own mirrored sphere so both subunits see the map. χ1/Rama are still **propagated**
    from master to copies that miss the master's sphere. Use ``--no-ncs-mirror-zones`` for the old
    single joint filter over all chains (only useful if copies overlap the master's spheres).

    See ``docs/ZONAL_GLOBAL_OVERLAP_AND_GMM.md``.
    """
    st = gemmi.read_structure(str(pdb))
    if len(st) == 0:
        raise typer.BadParameter("Empty structure.")
    try:
        st.merge_chain_parts()
    except Exception:
        pass

    master, copies = parse_ncs_chains(ncs)

    mv = read_map(map_path)
    progress_cb = None if quiet else (lambda m: typer.secho(m, err=True))
    g_result = run_global_zonal_refine(
        st,
        mv,
        pdb_path=Path(pdb),
        master_chain=master,
        copy_chains=copies,
        target_residues_per_region=target_residues_per_region,
        gmm_components=gmm_components,
        soft_resp_floor=soft_resp_floor,
        radius_pad=radius_pad,
        max_rounds=max_rounds,
        converge_rmsd_eps=converge_rmsd_eps,
        converge_patience=converge_patience,
        random_seed=random_seed,
        sse_from_pdb_header=sse_header,
        gmm_reg_covar=gmm_reg_covar,
        ncs_mirror_zones=ncs_mirror_zones,
        progress=progress_cb,
        passes=passes,
        weight_map=weight_map,
        map_density_threshold=map_density_threshold,
        weight_density_anchor=weight_density_anchor,
        weight_density_gain=weight_density_gain,
        map_anchor_eps=map_anchor_eps,
        weight_rot=weight_rot,
        soft_buffer=soft_buffer,
        soft_passes=soft_passes,
        soft_min_clash=soft_min_clash,
        soft_only_if_worsened=soft_only_worsened,
        rama_backbone=rama_backbone,
        rama_step_deg=rama_step_deg,
        rama_max_shift_deg=rama_max_shift_deg,
        weight_rama=weight_rama,
        weight_bb_move=weight_bb_move,
        rama_include_soft=rama_include_soft,
        rama_nudge_favored=rama_nudge_favored,
    )

    out_p = Path(out).expanduser()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_p))

    typer.echo(
        json.dumps(
            {
                "rounds_done": g_result.rounds_done,
                "stopped_reason": g_result.stopped_reason,
                "region_count": g_result.region_count,
                "final_ca_rmsd_vs_initial": round(g_result.final_ca_rmsd_vs_initial, 6)
                if math.isfinite(g_result.final_ca_rmsd_vs_initial)
                else None,
                "elapsed_sec": round(g_result.elapsed_sec, 3),
                "out_pdb": str(out_p),
            },
            indent=2,
        )
    )

    if json_log:
        write_global_result_json(json_log, g_result)
