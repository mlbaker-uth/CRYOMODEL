# crymodel/cli/findligands.py
from __future__ import annotations
from pathlib import Path
import typer
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from ..finders.pipeline import run_phase_1_to_3, write_phase_outputs
from ..finders.phase45 import (
    gate_points_by_distance,
    cluster_mean_centers,
    cluster_single_linkage,
    greedy_pseudoatoms_per_ligand_component,
    gate_ligand_components_by_distance,
)
from ..io.mrc import MapVolume
from ..io.pdb import read_model_xyz
from ..io.site_export import AssignmentSet, write_water_pdb, write_json, write_grouped_pseudoatoms_pdb, write_sites_csv
from ..io.mrc import read_map

app = typer.Typer(no_args_is_help=True)

@app.command()
def findligands(
    map: str = typer.Option(..., help="Input map (.mrc/.map)"),
    model: str = typer.Option(..., help="Model (PDB/mmCIF) for masking"),
    thresh: float = typer.Option(0.5, help="Map threshold after masking"),
    mask_radius: float = typer.Option(2.0, help="Mask radius around model (Å)"),
    micro_vvox_min: int = typer.Option(2, help="Minimum voxels for microblob"),
    micro_vvox_max: int = typer.Option(12, help="Maximum voxels for microblob"),
    zero_radius: float = typer.Option(2.0, help="Zero radius for greedy water picking (Å)"),
    # Phase 4: water gating + clustering
    water_gate_min: float = typer.Option(2.0, help="Min Å to nearest model atom (waters)"),
    water_gate_max: float = typer.Option(6.0, help="Max Å to nearest model atom (waters)"),
    water_cluster_radius: float = typer.Option(2.4, help="Clustering radius for waters (Å)"),
    # Phase 5: ligand pseudoatoms + gating
    ligand_zero_radius: float = typer.Option(1.5, help="Zero radius for ligand pseudoatoms (Å)"),
    ligand_gate_min: float = typer.Option(2.0, help="Min Å to nearest model atom (ligands)"),
    ligand_gate_max: float = typer.Option(10.0, help="Max Å to nearest model atom (ligands)"),
    half1: str = typer.Option(None, help="Optional half-map 1 (for stats)"),
    half2: str = typer.Option(None, help="Optional half-map 2 (for stats)"),
    ml_model: str = typer.Option(None, help="Optional path to trained ML model for prediction"),
    entry_resolution: float = typer.Option(None, help="Optional resolution for ML features (Å)"),
    keep_hydrogens: bool = typer.Option(False, help="Keep hydrogen atoms in model (default: remove them)"),
    out_dir: str = typer.Option("outputs", help="Output directory"),
):
    remove_hydrogens = not keep_hydrogens
    vol, masked, masked_thr, wmap, lmap, assigns = run_phase_1_to_3(
        map, model, thresh, mask_radius, micro_vvox_min, micro_vvox_max, zero_radius,
        half1_path=half1, half2_path=half2, remove_hydrogens=remove_hydrogens
    )
    write_phase_outputs(vol, masked, masked_thr, wmap, lmap, assigns, Path(out_dir))
    # Phase 4: distance gate + cluster waters -> candidate-waters
    mdl_xyzA = read_model_xyz(model, remove_hydrogens=remove_hydrogens)
    gated = gate_points_by_distance(assigns.centers_xyzA, mdl_xyzA, water_gate_min, water_gate_max)
    clustered = cluster_mean_centers(gated, water_cluster_radius)
    cand_assigns = AssignmentSet(centers_xyzA=clustered, meta={**assigns.meta,
        "water_gate_min_A": float(water_gate_min),
        "water_gate_max_A": float(water_gate_max),
        "water_cluster_radius_A": float(water_cluster_radius),
        "n_waters_candidate": int(clustered.shape[0]),
    })
    write_water_pdb(cand_assigns, Path(out_dir) / "candidate-waters")

    # Phase 5: ligand pseudoatoms per component + distance gating
    lig_vol = MapVolume(data_zyx=np.asarray(lmap, dtype=np.float32), apix=vol.apix, origin_xyzA=vol.origin_xyzA)
    comps_xyzA_all = greedy_pseudoatoms_per_ligand_component(lig_vol, ligand_zero_radius)
    # Track which components passed gating (by index in original list)
    comps_xyzA = gate_ligand_components_by_distance(comps_xyzA_all, mdl_xyzA, ligand_gate_min, ligand_gate_max)
    # Map original component index to pseudoatom count
    comp_idx_to_count = {i+1: len(comp) for i, comp in enumerate(comps_xyzA_all)}
    write_grouped_pseudoatoms_pdb(Path(out_dir) / "ligands", comps_xyzA)

    # Write debug json of parameters and counts
    debug_meta = {**cand_assigns.meta,
        "ligand_zero_radius_A": float(ligand_zero_radius),
        "ligand_gate_min_A": float(ligand_gate_min),
        "ligand_gate_max_A": float(ligand_gate_max),
        "n_ligand_components": int(len(comps_xyzA)),
    }
    write_json(AssignmentSet(centers_xyzA=cand_assigns.centers_xyzA, meta=debug_meta), Path(out_dir) / "assigns.json")

    # Build sites.csv with features
    rows = []
    # KD-tree for nearest protein atoms
    mdl_tree = cKDTree(np.asarray(mdl_xyzA, dtype=np.float32)) if mdl_xyzA.size else None

    # Get water cluster info for cluster_size and cluster_radius
    water_clusters = cluster_single_linkage(gated, water_cluster_radius) if gated.size > 0 else []
    # Build KD-tree of cluster means to match candidate centers to clusters
    cluster_means = np.array([c.mean(axis=0) for c in water_clusters], dtype=np.float32) if water_clusters else np.zeros((0,3), dtype=np.float32)
    cluster_tree = cKDTree(cluster_means) if len(cluster_means) > 0 else None

    # Water candidate rows
    for i, (x, y, z) in enumerate(np.asarray(cand_assigns.centers_xyzA, dtype=float), start=1):
        center = np.array([x, y, z], dtype=np.float32)
        min_dist = float(mdl_tree.query(center, k=1)[0]) if mdl_tree is not None else float("nan")
        # Find nearby protein atoms within 5Å
        nearby = mdl_tree.query_ball_point(center, r=5.0) if mdl_tree is not None else []
        n_nearby = len(nearby) if nearby else 0
        # Compute mean distance to nearby atoms
        if nearby and len(nearby) > 0:
            dists = np.linalg.norm(mdl_tree.data[nearby] - center, axis=1)
            mean_dist = float(dists.mean())
        else:
            mean_dist = float("nan")
        # Match cluster: find which cluster's mean is closest to this center
        if cluster_tree is not None:
            _, closest_cluster_idx = cluster_tree.query(center, k=1)
            cluster_size = len(water_clusters[closest_cluster_idx])
        else:
            cluster_size = 1
        cluster_radius_A = water_cluster_radius
        rows.append({
            "id": f"W{i:05d}",
            "type": "water_candidate",
            "center_x": x, "center_y": y, "center_z": z,
            "nvox": "", "peak": "", "mean": "",
            "min_dist_A": min_dist,
            "n_nearby_protein_atoms": n_nearby,
            "mean_protein_distance_A": mean_dist,
            "cluster_size": cluster_size,
            "cluster_radius_A": cluster_radius_A,
            "std_dev": "", "skewness": "", "local_max_count": "",
            "elongation": "", "sphericity": "",
        })

    # Ligand component rows with stats from ligand map
    lab, nlab = label(np.asarray(lmap) > 0.0)
    
    for ci in range(1, nlab+1):
        sel = (lab == ci)
        nvox = int(sel.sum())
        if nvox == 0:
            continue
        vals = np.asarray(lmap, dtype=np.float32)[sel]
        peak = float(vals.max())
        mean = float(vals.mean())
        std_dev = float(vals.std())
        # Skewness: approximate from mean, median, std
        median = float(np.median(vals))
        skewness = float(3.0 * (mean - median) / std_dev) if std_dev > 1e-6 else 0.0
        # Local max count = number of greedy pseudoatoms (ci corresponds to index ci-1 in comps_xyzA_all)
        local_max_count = comp_idx_to_count.get(ci, 0)
        # center-of-mass in voxel space then to Å
        zyx = np.array(np.nonzero(sel)).T.astype(np.float32)
        weights = vals.astype(np.float32)
        wsum = float(weights.sum()) if weights.size else 1.0
        com_zyx = (zyx * weights[:, None]).sum(axis=0) / max(wsum, 1e-8)
        # convert to Å
        z, y, x = com_zyx.tolist()
        cx = float(vol.origin_xyzA[0] + x * float(vol.apix))
        cy = float(vol.origin_xyzA[1] + y * float(vol.apix))
        cz = float(vol.origin_xyzA[2] + z * float(vol.apix))
        center = np.array([cx, cy, cz], dtype=np.float32)
        min_dist = float(mdl_tree.query(center, k=1)[0]) if mdl_tree is not None else float("nan")
        # Nearby protein atoms
        nearby = mdl_tree.query_ball_point(center, r=5.0) if mdl_tree is not None else []
        n_nearby = len(nearby) if nearby else 0
        if nearby and len(nearby) > 0:
            dists = np.linalg.norm(mdl_tree.data[nearby] - center, axis=1)
            mean_protein_dist = float(dists.mean())
        else:
            mean_protein_dist = float("nan")
        # Geometry: elongation and sphericity from blob shape
        # Principal axes via SVD of voxel coordinates (centered)
        zyx_centered = zyx - com_zyx
        if len(zyx_centered) > 3:
            _, s, _ = np.linalg.svd(zyx_centered, full_matrices=False)
            # elongation = ratio of longest to shortest axis
            elongation = float(s[0] / s[-1]) if s[-1] > 1e-6 else float("nan")
            # sphericity = 4π * (3V/4π)^(2/3) / A, approximate from eigenvalues
            # Simplified: ratio of volume-equivalent sphere surface to actual surface approximation
            # Use: sphericity ≈ (36π * V^2)^(1/3) / (sum of eigenvalues), simplified
            # For now, approximate from eigenvalues
            sphericity = float((36.0 * np.pi * (nvox**2))**(1/3) / (s.sum() + 1e-6)) if s.sum() > 1e-6 else 0.0
        else:
            elongation = float("nan")
            sphericity = float("nan")
        rows.append({
            "id": f"L{ci:05d}",
            "type": "ligand_component",
            "center_x": cx, "center_y": cy, "center_z": cz,
            "nvox": nvox, "peak": peak, "mean": mean,
            "min_dist_A": min_dist,
            "std_dev": std_dev,
            "skewness": skewness,
            "local_max_count": local_max_count,
            "elongation": elongation,
            "sphericity": sphericity,
            "n_nearby_protein_atoms": n_nearby,
            "mean_protein_distance_A": mean_protein_dist,
            "cluster_size": "", "cluster_radius_A": "",
        })

    # Optional half-map stats: sample mean at site centers (simple point sample)
    if half1 and half2:
        try:
            h1 = read_map(half1).data_zyx
            h2 = read_map(half2).data_zyx
            apix = float(vol.apix); ox, oy, oz = [float(v) for v in vol.origin_xyzA]
            def sample(a, x, y, z):
                ix = int(round((x - ox) / apix)); iy = int(round((y - oy) / apix)); iz = int(round((z - oz) / apix))
                if 0 <= iz < a.shape[0] and 0 <= iy < a.shape[1] and 0 <= ix < a.shape[2]:
                    return float(a[iz, iy, ix])
                return float("nan")
            for r in rows:
                x = float(r["center_x"]); y = float(r["center_y"]); z = float(r["center_z"])
                r["half1_val"] = sample(h1, x, y, z)
                r["half2_val"] = sample(h2, x, y, z)
        except Exception:
            pass

    write_sites_csv(Path(out_dir) / "sites.csv", rows)

    # Phase 6: Optional ML prediction for water candidates
    if ml_model and Path(ml_model).exists():
        try:
            from ..ml.predict import predict_water_identities
            # Check if this is an ensemble (directory) or single model
            is_ensemble_dir = Path(ml_model).is_dir()
            if is_ensemble_dir:
                typer.echo("Running ensemble ML prediction on water candidates...")
            else:
                typer.echo("Running ML prediction on water candidates...")
            preds_df = predict_water_identities(
                candidate_waters_pdb=Path(out_dir) / "candidate-waters.pdb",
                model_pdb=model,
                model_checkpoint=ml_model,
                sites_csv=Path(out_dir) / "sites.csv",
                entry_resolution=entry_resolution,
                water_map=Path(out_dir) / "waters_map.mrc",
                half1_map=half1 if half1 else None,
                half2_map=half2 if half2 else None,
                remove_hydrogens=remove_hydrogens,
                is_ensemble=is_ensemble_dir,
                output_csv=Path(out_dir) / "water-predictions.csv",
            )
            typer.echo(f"ML predictions complete. Wrote {len(preds_df)} predictions to {out_dir}/water-predictions.csv")
        except Exception as e:
            typer.echo(f"Warning: ML prediction failed: {e}", err=True)

    typer.echo(f"Done. Waters(raw): {assigns.meta['n_waters']} candidate: {cand_assigns.meta['n_waters_candidate']}. Ligand comps: {len(comps_xyzA)}. Wrote outputs to {out_dir}.")

if __name__ == "__main__":
    app()
