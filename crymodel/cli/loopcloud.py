# crymodel/cli/loopcloud.py
"""CLI command for loopcloud - loop completion."""
from __future__ import annotations
from pathlib import Path
import typer
import gemmi

from ..loops.loopcloud import (
    parse_anchor_spec,
    generate_loop_candidates,
    score_loops,
)
from ..io.mrc import read_map

app = typer.Typer(no_args_is_help=True)


@app.command()
def generate(
    model: str = typer.Option(..., "--model", help="Input model PDB/mmCIF"),
    anchors: str = typer.Option(..., "--anchors", help="Anchor specification (e.g., 'chainA:res123 -> chainA:res140')"),
    sequence: str = typer.Option(..., "--sequence", help="Sequence for missing residues"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
    map: str = typer.Option(None, "--map", help="Full map (.mrc) for density scoring"),
    half1: str = typer.Option(None, "--half1", help="Half-map 1 (.mrc)"),
    half2: str = typer.Option(None, "--half2", help="Half-map 2 (.mrc)"),
    num_candidates: int = typer.Option(50, "--num-candidates", help="Number of loop candidates to generate"),
    top_n: int = typer.Option(10, "--top-n", help="Number of top candidates to output"),
    ss_type: str = typer.Option("loop", "--ss-type", help="Secondary structure type: 'helix', 'sheet', or 'loop'"),
):
    """Generate clash-free loop completions."""
    model_path = Path(model)
    if not model_path.exists():
        typer.echo(f"ERROR: Model not found: {model_path}", err=True)
        raise typer.Exit(1)
    
    # Load structure
    structure = gemmi.read_structure(str(model_path))
    
    # Parse anchors
    start_chain, start_res, end_chain, end_res = parse_anchor_spec(anchors)
    
    # Load maps if provided
    map_vol = None
    half1_vol = None
    half2_vol = None
    
    if map:
        from ..io.mrc import read_map
        map_vol = read_map(map)
        if half1:
            half1_vol = read_map(half1)
        if half2:
            half2_vol = read_map(half2)
    
    # Generate candidates
    typer.echo(f"Generating {num_candidates} loop candidates...")
    candidates = generate_loop_candidates(
        structure, start_chain, start_res, end_chain, end_res,
        sequence, num_candidates, ss_type
    )
    
    typer.echo(f"Generated {len(candidates)} candidates")
    
    # Score candidates
    typer.echo("Scoring candidates...")
    scores_df = score_loops(candidates, structure, map_vol, half1_vol, half2_vol)
    
    # Write outputs
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    scores_path = out_path / "scores.csv"
    scores_df.to_csv(scores_path, index=False)
    typer.echo(f"Wrote: {scores_path}")
    
    # Write top N candidates as PDB
    top_candidates = scores_df.head(top_n)
    typer.echo(f"\nTop {top_n} candidates:")
    
    for idx, row in top_candidates.iterrows():
        candidate_id = int(row['candidate_id'])
        ca_positions = candidates[candidate_id]
        
        # Write PDB
        pdb_path = out_path / f"loopcloud_{candidate_id:03d}.pdb"
        with open(pdb_path, 'w') as f:
            serial = 1
            for ca_pos in ca_positions:
                f.write(
                    f"HETATM{serial:5d}  CA  LOOP X{serial:4d}    "
                    f"{ca_pos[0]:8.3f}{ca_pos[1]:8.3f}{ca_pos[2]:8.3f}  1.00 20.00           C  \n"
                )
                serial += 1
        
        typer.echo(f"  Candidate {candidate_id}: score={row['total_score']:.3f}, "
                   f"clashes={int(row['clash_count'])}, CC={row['cc_mask']:.3f}")
        typer.echo(f"    Wrote: {pdb_path}")


if __name__ == "__main__":
    app()

