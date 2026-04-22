/**
 * Shared workflow card SPECS — used by dna_workflow_ui_demo.html and cryomodel.html (V2 library).
 */
(function (global) {
const SPECS = {
  mapfilter_apply: {
    display_name: "Map filter",
    description: "Pre-process map (lowpass, Gaussian, threshold, etc.) for DNA axis / build",
    command_template:
      'cryomodel mapfilter apply "{input_map}" "{output_map}" --filter "{filter_type}" {resolution_arg} {sigma_arg} {threshold_arg} {low_res_arg} {high_res_arg} {butterworth_order_arg} {sharpen_arg}',
    inputs: [
      { id: "input_map", label: "Input map (.mrc)", required: true, artifact_type: "map.mrc" }
    ],
    params: [
      {
        id: "filter_type",
        label: "Filter type *",
        type: "string",
        required: true,
        default: "gaussian",
        options: [
          "gaussian",
          "lowpass",
          "highpass",
          "bandpass",
          "threshold",
          "binary",
          "laplacian",
          "laplacian-sharpen",
          "median",
          "bilateral",
          "butterworth-lowpass",
          "butterworth-highpass",
          "normalize"
        ]
      },
      { id: "resolution", label: "Resolution Å (low/high/butterworth)", type: "float", required: false, default: null, min: 0.5, max: 100.0 },
      { id: "sigma_vox", label: "Sigma voxels (gaussian / median size hint)", type: "float", required: false, default: null, min: 0.01, max: 50.0 },
      { id: "threshold", label: "Threshold (threshold / binary)", type: "float", required: false, default: null, min: 0.0, max: 1e6 },
      { id: "low_res", label: "Bandpass low-res cutoff (Å)", type: "float", required: false, default: null, min: 0.5, max: 200.0 },
      { id: "high_res", label: "Bandpass high-res cutoff (Å)", type: "float", required: false, default: null, min: 0.5, max: 200.0 },
      { id: "butterworth_order", label: "Butterworth order (optional)", type: "int", required: false, default: null, min: 1, max: 16 },
      { id: "sharpen_strength", label: "Sharpen strength (laplacian-sharpen)", type: "float", required: false, default: null, min: 0.01, max: 10.0 }
    ],
    outputs: [
      { id: "output_map", label: "Filtered map", default: "outputs/mapfilter/map_filtered.mrc", artifact_type: "map.mrc" }
    ],
    param_arg_builders: {
      resolution_arg: { when_param_present: "resolution", value_template: "--resolution {resolution}" },
      sigma_arg: { when_param_present: "sigma_vox", value_template: "--sigma-vox {sigma_vox}" },
      threshold_arg: { when_param_present: "threshold", value_template: "--threshold {threshold}" },
      low_res_arg: { when_param_present: "low_res", value_template: "--low-res {low_res}" },
      high_res_arg: { when_param_present: "high_res", value_template: "--high-res {high_res}" },
      butterworth_order_arg: { when_param_present: "butterworth_order", value_template: "--butterworth-order {butterworth_order}" },
      sharpen_arg: { when_param_present: "sharpen_strength", value_template: "--sharpen-strength {sharpen_strength}" }
    }
  },
  model2map_convert: {
    display_name: "Model to Map",
    description: "Convert model (PDB/mmCIF) to synthetic map at target resolution",
    command_template:
      'cryomodel model2map --model "{model}" --output-map "{output_map}" --resolution {resolution} --apix {apix} {box_arg} {center_arg} {occ_arg} {bfac_arg} {origin_arg}',
    inputs: [
      { id: "model", label: "Input model (PDB/mmCIF)", required: true, artifact_type: "model.structure" }
    ],
    params: [
      { id: "resolution", label: "Resolution (A)", type: "float", required: true, default: 3.0, min: 0.5, max: 50.0 },
      { id: "apix", label: "Sampling apix (A/px)", type: "float", required: true, default: 1.0, min: 0.1, max: 10.0 },
      { id: "box_vox", label: "Box size (vox, 0=auto)", type: "int", required: true, default: 0, min: 0, max: 8192 },
      { id: "center_mode", label: "Coordinate frame mode", type: "string", required: true, default: "--no-center", options: ["--no-center", "--center"] },
      { id: "occupancy_mode", label: "Scale by occupancy", type: "string", required: true, default: "--no-scale-occupancy", options: ["--scale-occupancy", "--no-scale-occupancy"] },
      { id: "bfactor_mode", label: "Scale by B-factor", type: "string", required: true, default: "--no-scale-bfactor", options: ["--scale-bfactor", "--no-scale-bfactor"] },
      { id: "origin_mode", label: "Origin convention", type: "string", required: true, default: "auto", options: ["auto", "half-box-shift", "zero"] }
    ],
    outputs: [
      { id: "output_map", label: "Output synthetic map", default: "outputs/model2map/model_density.mrc", artifact_type: "map.mrc" }
    ],
    param_arg_builders: {
      box_arg: { when_param_present: "box_vox", value_template: "--box {box_vox}" },
      center_arg: { when_param_present: "center_mode", value_template: "{center_mode}" },
      occ_arg: { when_param_present: "occupancy_mode", value_template: "{occupancy_mode}" },
      bfac_arg: { when_param_present: "bfactor_mode", value_template: "{bfactor_mode}" },
      origin_arg: { when_param_present: "origin_mode", value_template: "--origin-mode {origin_mode}" }
    }
  },
  affilter_run: {
    display_name: "AF Filter",
    description: "Filter AlphaFold model and identify domains",
    command_template:
      'cryomodel affilter "{input_pdb}" --output "{output_pdb}" --plddt-threshold {plddt_threshold} --filter-loops --filter-connectivity --out-dir "{out_dir}"',
    inputs: [
      { id: "input_pdb", label: "Input AlphaFold model (PDB)", required: true, artifact_type: "model.structure" }
    ],
    params: [
      { id: "plddt_threshold", label: "pLDDT threshold", type: "float", required: true, default: 0.5, min: 0.0, max: 1.0 }
    ],
    outputs: [
      { id: "output_pdb", label: "Filtered model PDB", default: "outputs/affilter/alphafold_filtered.pdb", artifact_type: "model.structure" },
      { id: "out_dir", label: "Output directory", default: "outputs/affilter", artifact_type: "dir" }
    ]
  },
  foldhunter_search: {
    display_name: "FoldHunter",
    description: "FFT cross-correlation map–model search (probe PDB uses same synthetic map as model2map)",
    command_template:
      'cryomodel foldhunter "{target_map}" --probe-pdb "{probe_pdb}" --resolution {resolution} --top-n {top_n} --out-dir "{out_dir}" --search-preset {search_preset} --rank-by {rank_by} {plddt_arg} {weight_plddt_arg} {seed_rotation_arg}',
    inputs: [
      { id: "target_map", label: "Target map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "probe_pdb", label: "Probe model PDB", required: true, artifact_type: "model.structure" }
    ],
    params: [
      { id: "resolution", label: "Resolution (Å)", type: "float", required: true, default: 3.0, min: 0.5, max: 20.0 },
      {
        id: "search_preset",
        label: "Search preset",
        type: "string",
        required: true,
        default: "fast",
        options: ["full", "fast", "refine"]
      },
      {
        id: "rank_by",
        label: "Rank finalists by",
        type: "string",
        required: true,
        default: "correlation",
        options: ["correlation", "inclusion", "combined"]
      },
      {
        id: "plddt_threshold",
        label: "pLDDT min (optional, 0–1; leave empty = off)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "weight_plddt_flag",
        label: "Weight probe map by pLDDT",
        type: "string",
        required: false,
        default: "",
        options: ["", "--weight-by-plddt"]
      },
      {
        id: "seed_rotation",
        label: "Seed quaternion w,x,y,z (refine preset only)",
        type: "string",
        required: false,
        default: ""
      },
      { id: "top_n", label: "Top candidates", type: "int", required: true, default: 10, min: 1, max: 200 }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/foldhunter", artifact_type: "dir" },
      { id: "best_fit_pdb", label: "Top fit PDB", default: "outputs/foldhunter/foldhunter_top_fit.pdb", artifact_type: "model.structure" },
      { id: "results_csv", label: "Results CSV", default: "outputs/foldhunter/foldhunter_results.csv", artifact_type: "table.csv" }
    ],
    param_arg_builders: {
      plddt_arg: { when_param_present: "plddt_threshold", value_template: "--plddt-threshold {plddt_threshold}" },
      weight_plddt_arg: { when_param_present: "weight_plddt_flag", value_template: "{weight_plddt_flag}" },
      seed_rotation_arg: { when_param_present: "seed_rotation", value_template: '--seed-rotation "{seed_rotation}"' }
    }
  },
  findligands_run: {
    display_name: "FindLigands",
    description: "Detect water/ligand candidate sites from map and model",
    command_template:
      'cryomodel findligands --map "{map}" --model "{model}" --thresh {thresh} --mask-radius {mask_radius} --out-dir "{out_dir}"',
    inputs: [
      { id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "model", label: "Model PDB", required: true, artifact_type: "model.structure" }
    ],
    params: [
      { id: "thresh", label: "Threshold", type: "float", required: true, default: 0.5, min: 0.0, max: 10.0 },
      { id: "mask_radius", label: "Mask radius (Å)", type: "float", required: true, default: 2.0, min: 0.5, max: 20.0 }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/findligands", artifact_type: "dir" },
      { id: "candidate_waters_pdb", label: "Candidate waters", default: "outputs/findligands/candidate-waters.pdb", artifact_type: "model.structure" },
      { id: "ligands_pdb", label: "Ligand pseudoatoms", default: "outputs/findligands/ligands.pdb", artifact_type: "model.structure" },
      { id: "ligand_map", label: "Ligand map", default: "outputs/findligands/ligands_map.mrc", artifact_type: "map.mrc" },
      { id: "sites_csv", label: "Sites CSV", default: "outputs/findligands/sites.csv", artifact_type: "table.csv" }
    ]
  },
  predictligands_run: {
    display_name: "PredictLigands",
    description: "Classify ligand components from findligands outputs",
    command_template:
      'cryomodel predictligands --ligands-pdb "{ligands_pdb}" --ligand-map "{ligand_map}" --model "{model}" --entry-resolution {entry_resolution} --out-dir "{out_dir}" --output-csv "{output_csv}"',
    inputs: [
      { id: "ligands_pdb", label: "Ligands PDB", required: true, artifact_type: "model.structure" },
      { id: "ligand_map", label: "Ligand map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "model", label: "Model PDB", required: true, artifact_type: "model.structure" }
    ],
    params: [
      { id: "entry_resolution", label: "Entry resolution (Å)", type: "float", required: true, default: 3.0, min: 0.5, max: 20.0 }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/predictligands", artifact_type: "dir" },
      { id: "output_csv", label: "Predictions CSV", default: "ligand-predictions.csv", artifact_type: "table.csv" }
    ]
  },
  fitprep_check: {
    display_name: "FitPrep",
    description: "Preflight checker for map-model alignment and quick diagnostics",
    command_template:
      'cryomodel fitprep --model "{model}" --map "{map}" --out-dir "{out_dir}"',
    inputs: [
      { id: "model", label: "Model PDB", required: true, artifact_type: "model.structure" },
      { id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" }
    ],
    params: [],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/fitprep", artifact_type: "dir" },
      { id: "report_json", label: "FitPrep report", default: "outputs/fitprep/fitprep_report.json", artifact_type: "json" }
    ]
  },
  validate_run: {
    display_name: "Validate",
    description:
      "Resolution-aware model validation (FitCheck): local CC vs Gaussian model, Q-lite, Ringer-lite, geometry stubs. " +
      "Optional **half maps** and **local resolution** map (e.g. cryoSPARC) must share the primary map voxel grid when used.",
    command_template:
      'cryomodel validate --model "{model}" --map "{map}" --out-dir "{out_dir}" {half1_arg} {half2_arg} {localres_arg}',
    inputs: [
      { id: "model", label: "Model PDB", required: true, artifact_type: "model.structure" },
      { id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "half1_map", label: "Half-map 1 (.mrc, optional)", required: false, artifact_type: "map.mrc" },
      { id: "half2_map", label: "Half-map 2 (.mrc, optional)", required: false, artifact_type: "map.mrc" },
      { id: "localres_map", label: "Local resolution map (.mrc, optional)", required: false, artifact_type: "map.mrc" }
    ],
    params: [],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/validate", artifact_type: "dir" },
      { id: "features_csv", label: "Features CSV", default: "outputs/validate/features.csv", artifact_type: "table.csv" }
    ],
    arg_builders: {
      half1_arg: { when_input_present: "half1_map", value_template: '--half1 "{half1_map}"' },
      half2_arg: { when_input_present: "half2_map", value_template: '--half2 "{half2_map}"' },
      localres_arg: { when_input_present: "localres_map", value_template: '--localres "{localres_map}"' }
    }
  },
  pathwalker2_discover: {
    display_name: "Pathwalker2",
    description: "Automatic trace discovery from density map",
    command_template:
      'cryomodel pathwalker2 --map "{map}" --threshold {threshold} --n-residues {n_residues} --out-dir "{out_dir}"',
    inputs: [
      { id: "map", label: "Input map (.mrc)", required: true, artifact_type: "map.mrc" }
    ],
    params: [
      { id: "threshold", label: "Threshold", type: "float", required: true, default: 0.05, min: 0.0, max: 10.0 },
      { id: "n_residues", label: "Estimated residues", type: "int", required: true, default: 300, min: 1, max: 200000 }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/pathwalker2", artifact_type: "dir" }
    ]
  },
  loopcloud_generate: {
    display_name: "LoopCloud",
    description: "Generate loop completion candidates between anchors",
    command_template:
      'cryomodel loopcloud --model "{model}" --anchors "{anchors}" --sequence "{sequence}" --out-dir "{out_dir}" {map_arg} {num_candidates_arg} {top_n_arg}',
    inputs: [
      { id: "model", label: "Input model (PDB/mmCIF)", required: true, artifact_type: "model.structure" },
      { id: "map", label: "Map (.mrc, optional for scoring)", required: false, artifact_type: "map.mrc" }
    ],
    params: [
      { id: "anchors", label: "Anchors spec", type: "string", required: true, default: "A:100 -> A:120" },
      { id: "sequence", label: "Missing loop sequence", type: "string", required: true, default: "AAAAA" },
      { id: "num_candidates", label: "Num candidates", type: "int", required: true, default: 50, min: 1, max: 2000 },
      { id: "top_n", label: "Top N output", type: "int", required: true, default: 10, min: 1, max: 500 }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/loopcloud", artifact_type: "dir" },
      { id: "scores_csv", label: "Scores CSV", default: "outputs/loopcloud/scores.csv", artifact_type: "table.csv" }
    ],
    arg_builders: {
      map_arg: { when_input_present: "map", value_template: '--map "{map}"' }
    },
    param_arg_builders: {
      num_candidates_arg: { when_param_present: "num_candidates", value_template: "--num-candidates {num_candidates}" },
      top_n_arg: { when_param_present: "top_n", value_template: "--top-n {top_n}" }
    }
  },
  fitcompare_run: {
    display_name: "FitCompare",
    description: "Align and compare two models",
    command_template:
      'cryomodel fitcompare --model-a "{model_a}" --model-b "{model_b}" --out-dir "{out_dir}"',
    inputs: [
      { id: "model_a", label: "Model A (reference)", required: true, artifact_type: "model.structure" },
      { id: "model_b", label: "Model B (to align)", required: true, artifact_type: "model.structure" }
    ],
    params: [],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/fitcompare", artifact_type: "dir" },
      { id: "superposed_pdb", label: "Superposed model", default: "outputs/fitcompare/fitcompare_superposed.pdb", artifact_type: "model.structure" },
      { id: "deltas_csv", label: "Per-residue deltas", default: "outputs/fitcompare/per_residue_deltas.csv", artifact_type: "table.csv" }
    ]
  },
  pdbdomain_identify: {
    display_name: "PDB Domain",
    description: "Identify structural domains from a model",
    command_template:
      'cryomodel pdbdomain --model "{model}" --out-prefix "{out_prefix}"',
    inputs: [
      { id: "model", label: "Model (PDB/mmCIF)", required: true, artifact_type: "model.structure" }
    ],
    params: [],
    outputs: [
      { id: "out_prefix", label: "Output prefix", default: "outputs/pdbdomain/domains", artifact_type: "other" },
      { id: "domains_json", label: "Domains JSON", default: "outputs/pdbdomain/domains.json", artifact_type: "json" },
      { id: "domains_csv", label: "Domains CSV", default: "outputs/pdbdomain/domains.csv", artifact_type: "table.csv" },
      { id: "domains_pdb", label: "Domains PDB", default: "outputs/pdbdomain/domains.pdb", artifact_type: "model.structure" }
    ]
  },
  pdbcom_compute: {
    display_name: "PDB COM",
    description: "Compute domain centers-of-mass from model and domain spec",
    command_template:
      'cryomodel pdbcom --model "{model}" --domains "{domains}" --out-prefix "{out_prefix}"',
    inputs: [
      { id: "model", label: "Model (PDB/mmCIF)", required: true, artifact_type: "model.structure" },
      { id: "domains", label: "Domains JSON", required: true, artifact_type: "json" }
    ],
    params: [],
    outputs: [
      { id: "out_prefix", label: "Output prefix", default: "outputs/pdbcom/domains_com", artifact_type: "other" },
      { id: "com_pdb", label: "COM PDB", default: "outputs/pdbcom/domains_com.pdb", artifact_type: "model.structure" },
      { id: "com_csv", label: "COM CSV", default: "outputs/pdbcom/domains_com.csv", artifact_type: "table.csv" }
    ]
  },
  dnaaxis_extract: {
    display_name: "DNA Axis",
    description: "Extract DNA centerline from map",
    command_template: 'cryomodel dnaaxis extract --map "{map}" --threshold {threshold} --out-pdb "{out_pdb}" --out-mrc "{out_mrc}" {guides_arg}',
    inputs: [
      { id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "guides_pdb", label: "Guides PDB (optional)", required: false, artifact_type: "model.structure" }
    ],
    params: [
      { id: "threshold", label: "Threshold", type: "float", required: true, default: 0.25, min: 0.0, max: 10.0 }
    ],
    outputs: [
      { id: "out_pdb", label: "Axis PDB", default: "outputs/dnaaxis/dna_axis.pdb", artifact_type: "model.structure" },
      { id: "out_mrc", label: "Axis MRC", default: "outputs/dnaaxis/dna_axis.mrc", artifact_type: "map.mrc" }
    ],
    arg_builders: {
      guides_arg: { when_input_present: "guides_pdb", value_template: '--guides-pdb "{guides_pdb}"' }
    }
  },
  dnabuild_build: {
    display_name: "Build DNA",
    description: "Poly-AT model from centerline (dnabuild build-2bp)",
    command_template:
      'cryomodel dnabuild build-2bp --centerline-pdb "{centerline_pdb}" --template-2bp-pdb "{template_2bp_pdb}" ' +
      '--map "{map}" --out-pdb "{out_pdb}" --target-spacing {target_spacing} {threshold_arg} {report_arg}',
    inputs: [
      { id: "centerline_pdb", label: "Centerline PDB", required: true, artifact_type: "model.structure" },
      { id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" }
    ],
    params: [
      {
        id: "template_2bp_pdb",
        label: "2-bp template PDB *",
        type: "string",
        required: true,
        default: "data/DNA-TEMPLATES/2AT-template.pdb"
      },
      { id: "target_spacing", label: "Target spacing (Å)", type: "float", required: true, default: 3.4, min: 2.0, max: 5.0 },
      { id: "threshold", label: "Threshold (optional, map/report)", type: "float", required: false, default: null, min: 0.0, max: 10.0 },
      { id: "report", label: "Report path (optional)", type: "string", required: false, default: null }
    ],
    outputs: [
      { id: "out_pdb", label: "Built DNA PDB", default: "outputs/dnabuild/dna_initial.pdb", artifact_type: "model.structure" }
    ],
    arg_builders: {
      threshold_arg: { when_input_present: "threshold", value_template: "--threshold {threshold}" }
    },
    param_arg_builders: {
      report_arg: { when_param_present: "report", value_template: '--report "{report}"' }
    }
  },
  basehunter_run: {
    display_name: "BaseHunter",
    description: "Classify bases and generate summary table",
    command_template: 'cryomodel basehunter --map "{map}" --model "{model}" --out-dir "{out_dir}" --resolution {resolution} {chain_arg}',
    inputs: [
      { id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "model", label: "Model PDB", required: true, artifact_type: "model.structure" }
    ],
    params: [
      { id: "resolution", label: "Resolution (Å)", type: "float", required: true, default: 3.0, min: 0.5, max: 20.0 },
      { id: "chain", label: "Chain (optional)", type: "string", required: false, default: null }
    ],
    outputs: [
      { id: "out_dir", label: "Output Directory", default: "outputs/basehunter", artifact_type: "dir" },
      { id: "scores_csv", label: "Scores CSV", default: "outputs/basehunter/basehunter_scores.csv", artifact_type: "table.csv" },
      { id: "summary_json", label: "Summary JSON", default: "outputs/basehunter/basehunter_summary.json", artifact_type: "json" }
    ],
    arg_builders: {
      chain_arg: { when_input_present: "chain", value_template: "--chain {chain}" }
    }
  },
  pathwalker_run: {
    display_name: "Pathwalker (legacy)",
    description: "Pseudoatom + TSP backbone trace in density (legacy engine; prefer Pathwalker2 for new maps)",
    command_template:
      'cryomodel pathwalker --map "{map}" --threshold {threshold} --n-residues {n_residues} --pseudoatom-method {pseudoatom_method} --tsp-solver {tsp_solver} --time-limit {time_limit} --random-state {random_state} --output-pdb "{output_pdb}" --out-dir "{out_dir}" {noise_arg}',
    inputs: [{ id: "map", label: "Map (.mrc)", required: true, artifact_type: "map.mrc" }],
    params: [
      { id: "threshold", label: "Threshold", type: "float", required: true, default: 0.05, min: 0.0, max: 10.0 },
      { id: "n_residues", label: "Residue count (Cα)", type: "int", required: true, default: 200, min: 1, max: 200000 },
      {
        id: "pseudoatom_method",
        label: "Pseudoatom method",
        type: "string",
        required: true,
        default: "kmeans",
        options: ["kmeans", "sc", "ac", "ms", "gmm", "birch"]
      },
      { id: "tsp_solver", label: "TSP solver", type: "string", required: true, default: "ortools", options: ["ortools", "lkh"] },
      { id: "time_limit", label: "TSP time limit (s)", type: "int", required: true, default: 30, min: 1, max: 86400 },
      { id: "random_state", label: "Random seed", type: "int", required: true, default: 42, min: 0, max: 2147483647 },
      { id: "noise_level", label: "Noise level Å (optional)", type: "float", required: false, default: null, min: 0.0, max: 10.0 }
    ],
    outputs: [
      { id: "output_pdb", label: "Output PDB", default: "outputs/pathwalker/pathwalker.pdb", artifact_type: "model.structure" },
      { id: "out_dir", label: "Output directory", default: "outputs/pathwalker", artifact_type: "dir" }
    ],
    param_arg_builders: {
      noise_arg: { when_param_present: "noise_level", value_template: "--noise-level {noise_level}" }
    }
  },
  pathwalker_average_run: {
    display_name: "Pathwalker average",
    description: "Average multiple legacy pathwalker PDB outputs",
    command_template: 'cryomodel pathwalker-average --path-files "{path_files}" --output-pdb "{output_pdb}" --out-dir "{out_dir}" {prob_flag}',
    inputs: [],
    params: [
      {
        id: "path_files",
        label: "Path PDB files (comma-separated)",
        type: "string",
        required: true,
        default: "outputs/pathwalker/run1.pdb,outputs/pathwalker/run2.pdb"
      },
      { id: "prob_flag", label: "Probabilistic B-factors", type: "string", required: false, default: "", options: ["", "--probabilistic"] }
    ],
    outputs: [
      { id: "output_pdb", label: "Averaged PDB", default: "outputs/pathwalker/pathwalker_averaged.pdb", artifact_type: "model.structure" },
      { id: "out_dir", label: "Output directory", default: "outputs/pathwalker", artifact_type: "dir" }
    ],
    param_arg_builders: {
      prob_flag: { when_param_present: "prob_flag", value_template: "{prob_flag}" }
    }
  },
  pyhole_analyze: {
    display_name: "PyHole",
    description: "Pore profile along top→bottom selections (non-interactive; set top and bottom)",
    command_template:
      'cryomodel pyhole analyze --pdb "{pdb}" --top "{top}" --bottom "{bottom}" --out-prefix "{out_prefix}" --step {step} --centerline {centerline}',
    inputs: [{ id: "pdb", label: "Structure PDB", required: true, artifact_type: "model.structure" }],
    params: [
      { id: "top", label: "Top selection (e.g. A:123)", type: "string", required: true, default: "A:1" },
      { id: "bottom", label: "Bottom selection", type: "string", required: true, default: "A:100" },
      { id: "out_prefix", label: "Output prefix", type: "string", required: true, default: "outputs/pyhole/pyhole_out" },
      { id: "step", label: "Step (Å)", type: "float", required: true, default: 1.0, min: 0.1, max: 50.0 },
      { id: "centerline", label: "Centerline", type: "string", required: true, default: "straight", options: ["straight", "curved"] }
    ],
    outputs: [
      { id: "out_dir", label: "Output dir (prefix parent)", default: "outputs/pyhole", artifact_type: "dir" }
    ]
  },
  pyhole_plot_run: {
    display_name: "PyHole plot",
    description: "Plot pyHole radius profiles from CSV/prefix outputs",
    command_template: 'cryomodel pyhole-plot plot "{inputs}" --out "{out}" {overlay_flag} {grid_arg}',
    inputs: [],
    params: [
      { id: "inputs", label: "Inputs (comma-separated prefixes or files)", type: "string", required: true, default: "outputs/pyhole/pyhole_out" },
      { id: "out", label: "Output basename", type: "string", required: true, default: "outputs/pyhole/pyhole_plot" },
      { id: "overlay_flag", label: "Overlay", type: "string", required: false, default: "", options: ["", "--overlay"] },
      { id: "grid", label: "Grid (e.g. 1x3)", type: "string", required: false, default: null }
    ],
    outputs: [],
    param_arg_builders: {
      overlay_flag: { when_param_present: "overlay_flag", value_template: "{overlay_flag}" },
      grid_arg: { when_param_present: "grid", value_template: '--grid "{grid}"' }
    }
  },
  train_ml_run: {
    display_name: "Train ML (ion/water)",
    description: "Train ion/water classifier from feature CSV (requires ML extras)",
    command_template:
      'cryomodel train-ml --train-csv "{train_csv}" --outdir "{outdir}" --epochs {epochs} --batch {batch} --lr {lr}',
    inputs: [],
    params: [
      { id: "train_csv", label: "Training CSV", type: "string", required: true, default: "data/features_train.csv" },
      { id: "outdir", label: "Output directory", type: "string", required: true, default: "outputs/train_ml/model" },
      { id: "epochs", label: "Epochs", type: "int", required: true, default: 40, min: 1, max: 10000 },
      { id: "batch", label: "Batch size", type: "int", required: true, default: 512, min: 1, max: 65536 },
      { id: "lr", label: "Learning rate", type: "float", required: true, default: 0.0002, min: 1e-8, max: 1.0 }
    ],
    outputs: [{ id: "outdir", label: "Model directory", default: "outputs/train_ml/model", artifact_type: "dir" }]
  },
  train_ensemble_run: {
    display_name: "Train ensemble",
    description: "Train an ensemble of ML models (requires ML extras)",
    command_template:
      'cryomodel train-ensemble --train-csv "{train_csv}" --outdir "{outdir}" --n-models {n_models} --epochs {epochs} --batch {batch} --lr {lr}',
    inputs: [],
    params: [
      { id: "train_csv", label: "Training CSV", type: "string", required: true, default: "data/features_train.csv" },
      { id: "outdir", label: "Output directory", type: "string", required: true, default: "outputs/train_ensemble/model" },
      { id: "n_models", label: "Number of models", type: "int", required: true, default: 3, min: 1, max: 32 },
      { id: "epochs", label: "Epochs per model", type: "int", required: true, default: 50, min: 1, max: 10000 },
      { id: "batch", label: "Batch size", type: "int", required: true, default: 512, min: 1, max: 65536 },
      { id: "lr", label: "Learning rate", type: "float", required: true, default: 0.0002, min: 1e-8, max: 1.0 }
    ],
    outputs: [{ id: "outdir", label: "Model directory", default: "outputs/train_ensemble/model", artifact_type: "dir" }]
  },
  extract_features_run: {
    display_name: "Extract features",
    description: "Extract ML features from a directory of PDBs (requires ML extras)",
    command_template: 'cryomodel extract-features --pdb-dir "{pdb_dir}" --output-csv "{output_csv}"',
    inputs: [],
    params: [
      { id: "pdb_dir", label: "PDB directory", type: "string", required: true, default: "data/pdb_subset" },
      { id: "output_csv", label: "Output CSV", type: "string", required: true, default: "outputs/extract_features/features.csv" }
    ],
    outputs: [{ id: "output_csv", label: "Features CSV", default: "outputs/extract_features/features.csv", artifact_type: "table.csv" }]
  },
  pathmeasure_launcher: {
    display_name: "PathMeasure Launcher",
    description: "Start/stop PathMeasure server and open the PathMeasure UI for tracing measurements",
    command_template: "pathmeasure-launch",
    inputs: [],
    params: [
      { id: "port", label: "Port", type: "int", required: true, default: 8008, min: 1, max: 65535 }
    ],
    outputs: []
  },
  pdb_mutate_run: {
    display_name: "PDB mutate (sequence)",
    description:
      "Introduce substitutions using exactly one sequence source: Target FASTA (one record, or full-length with auth mapping) OR Alignment FASTA (two equal-length rows from an MSA; default records 0 and 1, template auto-detected, chain matched as subsequence). Optional map for rotamer scoring. Guide metrics (clash/density) are written to --json-log when set.",
    command_template:
      'cryomodel pdb-mutate run "{pdb}" "{out_pdb}" --chain "{chain}" {target_fasta_arg} {alignment_fasta_arg} {alignment_rows_arg} {map_arg} --weight-rotamer {weight_rotamer} --weight-map {weight_map} --density-sigma-k {density_sigma_k} {json_log_arg}',
    inputs: [
      { id: "pdb", label: "Structure PDB", required: true, artifact_type: "model.structure" },
      {
        id: "target_fasta",
        label: "Target sequence FASTA (one record)",
        required: false,
        artifact_type: "sequence.fasta"
      },
      {
        id: "alignment_fasta",
        label: "Alignment FASTA (MSA: pick two rows via indices below)",
        required: false,
        artifact_type: "sequence.fasta"
      }
    ],
    params: [
      { id: "out_pdb", label: "Output PDB", type: "string", required: true, default: "outputs/pdb_mutate/mutated.pdb" },
      { id: "chain", label: "Chain ID (comma-separated homomultimer)", type: "string", required: true, default: "A" },
      {
        id: "alignment_row_a",
        label: "Alignment FASTA: first row index (0-based)",
        type: "int",
        required: true,
        default: 0,
        min: 0,
        max: 9999
      },
      {
        id: "alignment_row_b",
        label: "Alignment FASTA: second row index (0-based)",
        type: "int",
        required: true,
        default: 1,
        min: 0,
        max: 9999
      },
      { id: "map", label: "Map MRC (optional, rotamer + density guide)", type: "string", required: false, default: "" },
      {
        id: "weight_rotamer",
        label: "Rotamer prior weight (-log P)",
        type: "float",
        required: true,
        default: 0.15,
        min: 0,
        max: 100
      },
      {
        id: "weight_map",
        label: "Map density weight (if map set)",
        type: "float",
        required: true,
        default: 0.5,
        min: 0,
        max: 100
      },
      {
        id: "density_sigma_k",
        label: "Guide: k in threshold = map_mean + k·map_std",
        type: "float",
        required: true,
        default: 1.0,
        min: 0,
        max: 20
      },
      {
        id: "json_log",
        label: "Guide metrics JSON (mutations, clash/density deltas); empty to skip",
        type: "string",
        required: false,
        default: "outputs/pdb_mutate/mutate_log.json"
      }
    ],
    outputs: [
      { id: "out_pdb", label: "Mutated PDB", default: "outputs/pdb_mutate/mutated.pdb", artifact_type: "model.structure" },
      { id: "json_log", label: "Guide metrics JSON", default: "outputs/pdb_mutate/mutate_log.json", artifact_type: "table.json" }
    ],
    arg_builders: {
      target_fasta_arg: {
        when_input_present: "target_fasta",
        value_template: '--target-fasta "{target_fasta}"'
      },
      alignment_fasta_arg: {
        when_input_present: "alignment_fasta",
        value_template: '--alignment-fasta "{alignment_fasta}"'
      },
      alignment_rows_arg: {
        when_input_present: "alignment_fasta",
        value_template:
          '--alignment-row-a {alignment_row_a} --alignment-row-b {alignment_row_b}'
      }
    },
    param_arg_builders: {
      map_arg: { when_param_present: "map", value_template: '--map "{map}"' },
      json_log_arg: { when_param_present: "json_log", value_template: '--json-log "{json_log}"' }
    }
  },
  seqconservation_run: {
    display_name: "Sequence conservation map",
    description:
      "CLI: cryomodel seqconservation. Map MSA columns onto one or more chains (comma-separated homomultimers with identical polymer sequences). " +
      "First FASTA record is the reference (may be longer than the model; extra residues are skipped until coordinates match in order). " +
      "Per-residue metrics: n_aa_types, p_nonref, entropy, mean_penalty, etc. " +
      "ChimeraX: color by B-factor (e.g. n_aa_types); occupancy second channel (e.g. p_nonref). Clear optional paths to skip JSON/PDB.",
    command_template:
      'cryomodel seqconservation "{pdb}" "{alignment_fasta}" --chain "{chain}" --out-csv "{out_csv}" {out_json_arg} {out_pdb_arg} --bfactor-metric "{bfactor_metric}" --occupancy-metric "{occupancy_metric}" {include_ref_arg}',
    inputs: [
      { id: "pdb", label: "Structure PDB", required: true, artifact_type: "model.structure" },
      { id: "alignment_fasta", label: "MSA FASTA (record 1 = reference for chain)", required: true, artifact_type: "sequence.fasta" }
    ],
    params: [
      { id: "chain", label: "Chain ID(s), comma-separated homomultimer", type: "string", required: true, default: "A" },
      { id: "out_csv", label: "Output CSV", type: "string", required: true, default: "outputs/seqconservation/conservation.csv" },
      { id: "out_json", label: "Output JSON (optional)", type: "string", required: false, default: "outputs/seqconservation/conservation.json" },
      { id: "out_pdb", label: "Output PDB with mapped columns (optional)", type: "string", required: false, default: "outputs/seqconservation/conservation_bfactor.pdb" },
      {
        id: "bfactor_metric",
        label: "Metric → B-factor (dropdown)",
        type: "string",
        required: true,
        default: "n_aa_types",
        options: [
          { value: "n_aa_types", label: "n_aa_types — distinct amino-acid types (1–20)" },
          { value: "p_nonref", label: "p_nonref — fraction ≠ reference (0–1)" },
          { value: "p_gap", label: "p_gap — fraction of sequences gapped" },
          { value: "entropy", label: "entropy — Shannon diversity of AA counts" },
          { value: "p_major", label: "p_major — frequency of dominant AA" },
          { value: "mean_penalty", label: "mean_penalty — avg chemical/size penalty vs ref" },
          { value: "frac_nonconservative", label: "frac_nonconservative — harsh substitution fraction" }
        ]
      },
      {
        id: "occupancy_metric",
        label: "Metric → occupancy (dropdown)",
        type: "string",
        required: true,
        default: "p_nonref",
        options: [
          { value: "p_nonref", label: "p_nonref — fraction ≠ reference (0–1)" },
          { value: "n_aa_types", label: "n_aa_types — distinct amino-acid types" },
          { value: "p_gap", label: "p_gap — gap fraction" },
          { value: "entropy", label: "entropy — Shannon diversity" },
          { value: "p_major", label: "p_major — dominant AA frequency" },
          { value: "mean_penalty", label: "mean_penalty — chemical penalty" },
          { value: "frac_nonconservative", label: "frac_nonconservative — harsh fraction" }
        ]
      },
      {
        id: "include_ref_flag",
        label: "Include reference sequence in frequency stats",
        type: "string",
        required: false,
        default: "",
        options: ["", "--include-reference-in-stats"]
      }
    ],
    outputs: [
      { id: "out_csv", label: "Conservation CSV", default: "outputs/seqconservation/conservation.csv", artifact_type: "table.csv" },
      { id: "out_json", label: "Conservation JSON", default: "outputs/seqconservation/conservation.json", artifact_type: "table.json" },
      { id: "out_pdb", label: "Conservation mapped PDB", default: "outputs/seqconservation/conservation_bfactor.pdb", artifact_type: "model.structure" },
      { id: "msa_fasta", label: "MSA FASTA path (for chaining)", default: "", artifact_type: "sequence.fasta", skip_run_version: true }
    ],
    output_passthrough: { msa_fasta: "alignment_fasta" },
    param_arg_builders: {
      out_json_arg: { when_param_present: "out_json", value_template: '--out-json "{out_json}"' },
      out_pdb_arg: { when_param_present: "out_pdb", value_template: '--out-pdb "{out_pdb}"' },
      include_ref_arg: { when_param_present: "include_ref_flag", value_template: "{include_ref_flag}" }
    }
  },
  seqconservation_diffuse_run: {
    display_name: "Sequence conservation — 3D diffusion (Tier 3)",
    description:
      "CLI: cryomodel seqconservation-diffuse. Same MSA→residue table as conservation map, then **3D Cα graph diffusion** across **all** listed chains (edges within and between subunits). " +
      "Choose seed metric from dropdown: **raw** column stats or **composite** (e.g. p_nonref×mean_penalty) to emphasize chemically strong variation. " +
      "CSV adds seed_raw, seed_signal, diffused_score, basin_id. Tune seed threshold (composites are smaller scale). ChimeraX: B-factor from diffused_score or seed_signal.",
    command_template:
      'cryomodel seqconservation-diffuse "{pdb}" "{alignment_fasta}" --chain "{chain}" --out-csv "{out_csv}" {out_json_arg} {out_pdb_arg} --seed-metric "{seed_metric}" --seed-threshold {seed_threshold} --contact-radius {contact_radius} --falloff-angstrom {falloff_angstrom} --diffusion-steps {diffusion_steps} --mix {mix} --peak-min {peak_min} --basin-mode "{basin_mode}" --peak-weight-gamma {peak_weight_gamma} --bfactor-writes "{bfactor_writes}" {include_ref_arg}',
    inputs: [
      { id: "pdb", label: "Structure PDB", required: true, artifact_type: "model.structure" },
      { id: "alignment_fasta", label: "MSA FASTA (record 1 = reference)", required: true, artifact_type: "sequence.fasta" }
    ],
    params: [
      { id: "chain", label: "Chain ID(s), comma-separated (one 3D graph)", type: "string", required: true, default: "A,B,C,D" },
      { id: "out_csv", label: "Output CSV", type: "string", required: true, default: "outputs/seqconservation/diffusion.csv" },
      { id: "out_json", label: "JSON meta+rows (optional)", type: "string", required: false, default: "outputs/seqconservation/diffusion.json" },
      { id: "out_pdb", label: "PDB B-factor = diffused or seed (optional)", type: "string", required: false, default: "outputs/seqconservation/diffusion_bfactor.pdb" },
      {
        id: "seed_metric",
        label: "Seed metric (raw vs composite)",
        type: "string",
        required: true,
        default: "composite_nonref_penalty",
        option_groups: [
          {
            label: "Raw (MSA-derived)",
            options: [
              { value: "p_nonref", label: "p_nonref — fraction ≠ reference" },
              { value: "n_aa_types", label: "n_aa_types — distinct AA types at column" },
              { value: "entropy", label: "entropy — Shannon diversity" },
              { value: "mean_penalty", label: "mean_penalty — avg penalty vs ref (alone)" },
              { value: "frac_nonconservative", label: "frac_nonconservative — harsh AA-change fraction" },
              { value: "p_gap", label: "p_gap — column gap frequency" }
            ]
          },
          {
            label: "Composite (penalty-aware)",
            options: [
              { value: "composite_nonref_penalty", label: "p_nonref × mean_penalty" },
              { value: "composite_entropy_noncons", label: "entropy × frac_nonconservative" },
              { value: "composite_diversity_penalty", label: "((n_aa_types−1)/19) × mean_penalty" }
            ]
          }
        ]
      },
      { id: "seed_threshold", label: "Subtract from seed before diffuse", type: "float", required: true, default: 0.0, min: 0, max: 10 },
      { id: "contact_radius", label: "Cα edge cutoff (Å)", type: "float", required: true, default: 10.0, min: 4, max: 30 },
      { id: "falloff_angstrom", label: "exp(-d/d0) falloff d0 (Å)", type: "float", required: true, default: 3.0, min: 1, max: 15 },
      { id: "diffusion_steps", label: "Diffusion iterations", type: "int", required: true, default: 24, min: 1, max: 500 },
      { id: "mix", label: "Blend to neighbor mean / step", type: "float", required: true, default: 0.4, min: 0.05, max: 1 },
      { id: "peak_min", label: "Min diffused value for local peak", type: "float", required: true, default: 0.02, min: 0, max: 50 },
      {
        id: "basin_mode",
        label: "Basin grouping",
        type: "string",
        required: true,
        default: "nearest_peak",
        options: ["nearest_peak", "none"]
      },
      { id: "peak_weight_gamma", label: "Basin score: dist / peak^γ", type: "float", required: true, default: 0.5, min: 0.1, max: 3 },
      {
        id: "bfactor_writes",
        label: "PDB B-factor source",
        type: "string",
        required: true,
        default: "diffused_score",
        options: [
          { value: "diffused_score", label: "diffused_score — after graph diffusion" },
          { value: "seed_signal", label: "seed_signal — after threshold (pre-diffuse)" }
        ]
      },
      {
        id: "include_ref_flag",
        label: "Include reference in frequency stats",
        type: "string",
        required: false,
        default: "",
        options: ["", "--include-reference-in-stats"]
      }
    ],
    outputs: [
      { id: "out_csv", label: "Diffusion CSV", default: "outputs/seqconservation/diffusion.csv", artifact_type: "table.csv" },
      { id: "out_json", label: "Diffusion JSON", default: "outputs/seqconservation/diffusion.json", artifact_type: "table.json" },
      { id: "out_pdb", label: "Diffusion PDB", default: "outputs/seqconservation/diffusion_bfactor.pdb", artifact_type: "model.structure" },
      { id: "msa_fasta", label: "MSA FASTA path (for chaining)", default: "", artifact_type: "sequence.fasta", skip_run_version: true }
    ],
    output_passthrough: { msa_fasta: "alignment_fasta" },
    param_arg_builders: {
      out_json_arg: { when_param_present: "out_json", value_template: '--out-json "{out_json}"' },
      out_pdb_arg: { when_param_present: "out_pdb", value_template: '--out-pdb "{out_pdb}"' },
      include_ref_arg: { when_param_present: "include_ref_flag", value_template: "{include_ref_flag}" }
    }
  },
  zonal_refine_run: {
    display_name: "Zonal refine — local (sphere)",
    description:
      "Single-sphere local refinement: χ1 in a fixed center/radius (map + clash + rotamer), optional soft shell, optional φ/ψ Ramachandran micro-moves. " +
      "Set sphere center and radius here. For iterative GMM-based coverage of the whole structure (master + NCS), use the separate **Zonal refine — global** card.",
    command_template:
      'cryomodel zonal-refine run "{pdb}" "{map_mrc}" "{out_pdb}" --center "{center}" --radius {radius} {chains_arg} --passes {passes} --weight-map {weight_map} --weight-rotamer {weight_rotamer} --map-density-threshold {map_density_threshold} --weight-density-anchor {weight_density_anchor} --weight-density-gain {weight_density_gain} --map-anchor-eps {map_anchor_eps} --soft-buffer {soft_buffer} --soft-passes {soft_passes} --soft-min-clash {soft_min_clash} {soft_any_clash_flag} {rama_flag} --rama-step-deg {rama_step_deg} --rama-max-shift-deg {rama_max_shift_deg} --weight-rama {weight_rama} --weight-backbone-move {weight_bb_move} {rama_include_soft_flag} {rama_nudge_favored_flag} {json_log_arg}',
    inputs: [
      { id: "pdb", label: "Input model (PDB/mmCIF)", required: true, artifact_type: "model.structure" },
      { id: "map_mrc", label: "Map (.mrc / same frame as model)", required: true, artifact_type: "map.mrc" }
    ],
    params: [
      {
        id: "out_pdb",
        label: "Output refined PDB",
        type: "string",
        required: true,
        default: "outputs/zonal_refine/refined.pdb"
      },
      {
        id: "center",
        label: "Sphere center x,y,z (Å, comma-separated)",
        type: "string",
        required: true,
        default: "0,0,0"
      },
      { id: "radius", label: "Hard zone radius (Å)", type: "float", required: true, default: 8.0, min: 0.1, max: 200.0 },
      { id: "chains", label: "Chains (comma-separated, empty = all)", type: "string", required: false, default: "" },
      { id: "passes", label: "χ1 passes (hard zone)", type: "int", required: true, default: 3, min: 1, max: 50 },
      { id: "weight_map", label: "Map weight", type: "float", required: true, default: 0.65, min: 0, max: 100 },
      {
        id: "weight_rotamer",
        label: "Rotamer prior weight",
        type: "float",
        required: true,
        default: 0.15,
        min: 0,
        max: 10
      },
      {
        id: "map_density_threshold",
        label: "Map density threshold (raw units; 0 = off)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0,
        max: 1e6
      },
      {
        id: "weight_density_anchor",
        label: "Density anchor weight (0 = off)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0,
        max: 100
      },
      {
        id: "weight_density_gain",
        label: "Density gain bonus weight (0 = off)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0,
        max: 100
      },
      {
        id: "map_anchor_eps",
        label: "Map anchor ε (in-density vs weak)",
        type: "float",
        required: true,
        default: 0.00001,
        min: 0,
        max: 100
      },
      { id: "soft_buffer", label: "Soft shell extra Å (0 = χ1 hard only)", type: "float", required: true, default: 0.0, min: 0, max: 100 },
      { id: "soft_passes", label: "Soft-shell χ1 passes", type: "int", required: true, default: 2, min: 1, max: 20 },
      { id: "soft_min_clash", label: "Soft: min clash to consider χ1", type: "float", required: true, default: 1.0, min: 0, max: 1e6 },
      {
        id: "soft_any_clash_flag",
        label: "Soft shell: clash gate",
        type: "string",
        required: true,
        default: "",
        options: ["", "--soft-any-clash"]
      },
      {
        id: "rama_flag",
        label: "Ramachandran backbone micro-moves",
        type: "string",
        required: true,
        default: "",
        options: ["", "--rama-backbone"]
      },
      { id: "rama_step_deg", label: "Rama: φ/ψ grid step (°)", type: "float", required: true, default: 3.0, min: 0.5, max: 15.0 },
      { id: "rama_max_shift_deg", label: "Rama: max |Δφ|/|Δψ| (°)", type: "float", required: true, default: 9.0, min: 0, max: 30.0 },
      { id: "weight_rama", label: "Rama prior weight", type: "float", required: true, default: 0.08, min: 0, max: 10 },
      {
        id: "weight_bb_move",
        label: "Rama: penalty on Δφ²+Δψ²",
        type: "float",
        required: true,
        default: 0.015,
        min: 0,
        max: 10
      },
      {
        id: "rama_include_soft_flag",
        label: "Rama: include soft-shell residues",
        type: "string",
        required: true,
        default: "",
        options: ["", "--rama-include-soft"]
      },
      {
        id: "rama_nudge_favored_flag",
        label: "Rama: allow moves when already favored",
        type: "string",
        required: true,
        default: "",
        options: ["", "--rama-nudge-favored"]
      },
      {
        id: "json_log",
        label: "JSON summary log (empty to skip)",
        type: "string",
        required: false,
        default: "outputs/zonal_refine/zonal_refine.json"
      }
    ],
    outputs: [
      {
        id: "out_pdb",
        label: "Refined PDB",
        default: "outputs/zonal_refine/refined.pdb",
        artifact_type: "model.structure"
      },
      {
        id: "json_log",
        label: "Run summary JSON",
        default: "outputs/zonal_refine/zonal_refine.json",
        artifact_type: "table.json"
      }
    ],
    param_arg_builders: {
      chains_arg: { when_param_present: "chains", value_template: '--chains "{chains}"' },
      json_log_arg: { when_param_present: "json_log", value_template: '--json-log "{json_log}"' }
    }
  },
  zonal_refine_global_run: {
    display_name: "Zonal refine — global (GMM + NCS)",
    description:
      "Iterative **global** refinement: fit overlapping 3D GMM zones on the master chain Cα cloud, run local zonal χ1 in each zone, optionally propagate to NCS copies. " +
      "Requires `--ncs` (master first, then copies). Local solver options mirror **Zonal refine — local**. CLI: `cryomodel zonal-refine global`.",
    command_template:
      'cryomodel zonal-refine global "{pdb}" "{map_mrc}" "{out_pdb}" --ncs "{ncs}" --target-residues-per-region {target_residues_per_region} {gmm_components_arg} --soft-resp-floor {soft_resp_floor} --radius-pad {radius_pad} --max-rounds {max_rounds} --converge-rmsd-eps {converge_rmsd_eps} --converge-patience {converge_patience} --random-seed {random_seed} {sse_no_header_flag} --gmm-reg-covar {gmm_reg_covar} {ncs_mirror_no_flag} --passes {passes} --weight-map {weight_map} --map-density-threshold {map_density_threshold} --weight-density-anchor {weight_density_anchor} --weight-density-gain {weight_density_gain} --map-anchor-eps {map_anchor_eps} --weight-rotamer {weight_rotamer} --soft-buffer {soft_buffer} --soft-passes {soft_passes} --soft-min-clash {soft_min_clash} {soft_any_clash_flag} {rama_flag} --rama-step-deg {rama_step_deg} --rama-max-shift-deg {rama_max_shift_deg} --weight-rama {weight_rama} --weight-backbone-move {weight_bb_move} {rama_include_soft_flag} {rama_nudge_favored_flag} {json_log_arg} {quiet_flag}',
    inputs: [
      { id: "pdb", label: "Input model (PDB/mmCIF)", required: true, artifact_type: "model.structure" },
      { id: "map_mrc", label: "Map (.mrc / same frame as model)", required: true, artifact_type: "map.mrc" }
    ],
    params: [
      {
        id: "out_pdb",
        label: "Output refined PDB",
        type: "string",
        required: true,
        default: "outputs/zonal_refine_global/refined.pdb"
      },
      {
        id: "ncs",
        label: "NCS chains (master first, then copies, e.g. A or A,B,C,D)",
        type: "string",
        required: true,
        default: "A"
      },
      {
        id: "target_residues_per_region",
        label: "Target residues per GMM region (K ≈ N_Cα / this)",
        type: "int",
        required: true,
        default: 30,
        min: 5,
        max: 500
      },
      {
        id: "gmm_components",
        label: "GMM components K (empty = auto from target residues/region)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "soft_resp_floor",
        label: "Min GMM posterior to assign residue to region",
        type: "float",
        required: true,
        default: 0.12,
        min: 0.01,
        max: 0.99
      },
      { id: "radius_pad", label: "Extra Å beyond farthest Cα in region", type: "float", required: true, default: 4.0, min: 0.5, max: 50.0 },
      { id: "max_rounds", label: "Macro-round cap", type: "int", required: true, default: 7, min: 1, max: 100 },
      {
        id: "converge_rmsd_eps",
        label: "Stop if master Cα RMSD change below (Å)",
        type: "float",
        required: true,
        default: 0.03,
        min: 1e-6,
        max: 10.0
      },
      { id: "converge_patience", label: "Consecutive rounds under RMSD threshold", type: "int", required: true, default: 2, min: 1, max: 20 },
      { id: "random_seed", label: "RNG seed (GMM init, shuffle)", type: "int", required: true, default: 0, min: 0, max: 2147483647 },
      {
        id: "sse_no_header_flag",
        label: "PDB HELIX/SHEET expansion",
        type: "string",
        required: true,
        default: "",
        options: ["", "--no-sse-header"]
      },
      {
        id: "gmm_reg_covar",
        label: "GMM reg_covar (sklearn stability)",
        type: "float",
        required: true,
        default: 0.0001,
        min: 1e-9,
        max: 1.0
      },
      {
        id: "ncs_mirror_no_flag",
        label: "NCS: mirror zones per copy",
        type: "string",
        required: true,
        default: "",
        options: ["", "--no-ncs-mirror-zones"]
      },
      { id: "passes", label: "χ1 passes per local zone", type: "int", required: true, default: 3, min: 1, max: 50 },
      { id: "weight_map", label: "Map weight", type: "float", required: true, default: 0.65, min: 0, max: 100 },
      {
        id: "weight_rotamer",
        label: "Rotamer prior weight",
        type: "float",
        required: true,
        default: 0.15,
        min: 0,
        max: 10
      },
      {
        id: "map_density_threshold",
        label: "Map density threshold (raw units; 0 = off)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0,
        max: 1e6
      },
      {
        id: "weight_density_anchor",
        label: "Density anchor weight (0 = off)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0,
        max: 100
      },
      {
        id: "weight_density_gain",
        label: "Density gain bonus weight (0 = off)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0,
        max: 100
      },
      {
        id: "map_anchor_eps",
        label: "Map anchor ε (in-density vs weak)",
        type: "float",
        required: true,
        default: 0.00001,
        min: 0,
        max: 100
      },
      { id: "soft_buffer", label: "Soft shell extra Å (0 = χ1 hard only)", type: "float", required: true, default: 0.0, min: 0, max: 100 },
      { id: "soft_passes", label: "Soft-shell χ1 passes", type: "int", required: true, default: 2, min: 1, max: 20 },
      { id: "soft_min_clash", label: "Soft: min clash to consider χ1", type: "float", required: true, default: 1.0, min: 0, max: 1e6 },
      {
        id: "soft_any_clash_flag",
        label: "Soft shell: clash gate",
        type: "string",
        required: true,
        default: "",
        options: ["", "--soft-any-clash"]
      },
      {
        id: "rama_flag",
        label: "Ramachandran backbone micro-moves",
        type: "string",
        required: true,
        default: "",
        options: ["", "--rama-backbone"]
      },
      { id: "rama_step_deg", label: "Rama: φ/ψ grid step (°)", type: "float", required: true, default: 3.0, min: 0.5, max: 15.0 },
      { id: "rama_max_shift_deg", label: "Rama: max |Δφ|/|Δψ| (°)", type: "float", required: true, default: 9.0, min: 0, max: 30.0 },
      { id: "weight_rama", label: "Rama prior weight", type: "float", required: true, default: 0.08, min: 0, max: 10 },
      {
        id: "weight_bb_move",
        label: "Rama: penalty on Δφ²+Δψ²",
        type: "float",
        required: true,
        default: 0.015,
        min: 0,
        max: 10
      },
      {
        id: "rama_include_soft_flag",
        label: "Rama: include soft-shell residues",
        type: "string",
        required: true,
        default: "",
        options: ["", "--rama-include-soft"]
      },
      {
        id: "rama_nudge_favored_flag",
        label: "Rama: allow moves when already favored",
        type: "string",
        required: true,
        default: "",
        options: ["", "--rama-nudge-favored"]
      },
      {
        id: "json_log",
        label: "Global summary JSON (empty to skip)",
        type: "string",
        required: false,
        default: "outputs/zonal_refine_global/global_refine.json"
      },
      {
        id: "quiet_flag",
        label: "Quiet (suppress progress on stderr)",
        type: "string",
        required: true,
        default: "",
        options: ["", "--quiet"]
      }
    ],
    outputs: [
      {
        id: "out_pdb",
        label: "Refined PDB",
        default: "outputs/zonal_refine_global/refined.pdb",
        artifact_type: "model.structure"
      },
      {
        id: "json_log",
        label: "Global run summary JSON",
        default: "outputs/zonal_refine_global/global_refine.json",
        artifact_type: "table.json"
      }
    ],
    param_arg_builders: {
      gmm_components_arg: { when_param_present: "gmm_components", value_template: "--gmm-components {gmm_components}" },
      json_log_arg: { when_param_present: "json_log", value_template: '--json-log "{json_log}"' }
    }
  },
  alignment_sequence_pick_run: {
    display_name: "Pick FASTA sequence (for mutate)",
    description:
      "Small bridge: **Run** reads the multi-FASTA path from Inputs and writes **one** gap-stripped record (CLI: `fasta-extract row`). " +
      "Use **Load** (file or paste) to list sequences by header, pick one, then set row if needed. " +
      "On **PDB mutate**, inherit **Target sequence FASTA** from this card’s **out_fasta** output (or use alignment mode there instead).",
    command_template:
      'cryomodel fasta-extract row "{alignment_fasta}" "{out_fasta}" --row {selected_row}',
    inputs: [
      {
        id: "alignment_fasta",
        label: "MSA / multi-FASTA path (inherit msa_fasta from conservation or manual)",
        required: true,
        artifact_type: "sequence.fasta"
      }
    ],
    params: [
      {
        id: "selected_row",
        label: "Record index to extract (0-based, synced from pick list)",
        type: "int",
        required: true,
        default: 0,
        min: 0,
        max: 99999
      },
      {
        id: "out_fasta",
        label: "Output single-record FASTA path",
        type: "string",
        required: true,
        default: "outputs/fasta_pick/selected.fasta"
      }
    ],
    outputs: [
      {
        id: "out_fasta",
        label: "Selected sequence FASTA",
        default: "outputs/fasta_pick/selected.fasta",
        artifact_type: "sequence.fasta"
      }
    ]
  },
  symmetry_find_run: {
    display_name: "Symmetry find (Cₙ / Dₙ)",
    description:
      "Full **point-symmetry** pipeline: preprocess → axis candidates → rotational self-correlation → optional phase-3 refine, multishell scoring, and axis trace PDB. " +
      "CLI: `cryomodel symmetry find`. Use **guided** mode with **guided order** when you already know n.",
    command_template:
      'cryomodel symmetry find "{input_map}" "{out_dir}" --downsample {downsample} --density-percentile {density_percentile} --edge {edge} --laplacian-strength {laplacian_strength} --tilt-deg "{tilt_deg}" --axial-bins {axial_bins} --family {family} --mode {mode} --seed {seed} --max-voxels-pca {max_voxels_pca} {mask_arg} {density_threshold_arg} {orders_arg} {guided_order_arg} {no_diagonals_flag} {no_phase3_flag} {no_multishell_flag} {no_axis_pdb_flag}',
    inputs: [
      { id: "input_map", label: "Input map (.mrc)", required: true, artifact_type: "map.mrc" },
      { id: "mask", label: "Optional mask map (.mrc)", required: false, artifact_type: "map.mrc" }
    ],
    params: [
      {
        id: "downsample",
        label: "Downsample factor (-d)",
        type: "int",
        required: true,
        default: 4,
        min: 1,
        max: 64
      },
      {
        id: "density_percentile",
        label: "Density percentile (voxel mask for PCA; ignored if threshold set)",
        type: "float",
        required: true,
        default: 90.0,
        min: 0.0,
        max: 100.0
      },
      {
        id: "density_threshold",
        label: "Absolute density threshold (optional; overrides percentile when set)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "edge",
        label: "Edge emphasis",
        type: "string",
        required: true,
        default: "none",
        options: ["none", "laplacian", "laplacian_sharpen"]
      },
      {
        id: "laplacian_strength",
        label: "Laplacian sharpen strength",
        type: "float",
        required: true,
        default: 1.0,
        min: 0.0,
        max: 10.0
      },
      {
        id: "tilt_deg",
        label: "Tilt angles (deg, comma-separated)",
        type: "string",
        required: true,
        default: "0,5,10,15"
      },
      {
        id: "axial_bins",
        label: "Axial profile bins",
        type: "int",
        required: true,
        default: 64,
        min: 8,
        max: 512
      },
      {
        id: "family",
        label: "Symmetry family",
        type: "string",
        required: true,
        default: "c",
        options: ["c", "d", "auto"]
      },
      {
        id: "mode",
        label: "Search mode",
        type: "string",
        required: true,
        default: "search",
        options: ["search", "guided"]
      },
      {
        id: "guided_order",
        label: "Guided order n (required when mode is guided)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "orders",
        label: "Orders to score (comma-separated, optional; default is family-specific)",
        type: "string",
        required: false,
        default: ""
      },
      { id: "seed", label: "Random seed (PCA subsampling)", type: "int", required: true, default: 0, min: 0, max: 2_147_483_647 },
      {
        id: "max_voxels_pca",
        label: "Max voxels for weighted PCA",
        type: "int",
        required: true,
        default: 400000,
        min: 1000,
        max: 10_000_000
      },
      {
        id: "no_diagonals_flag",
        label: "Axis candidates",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-diagonals"]
      },
      {
        id: "no_phase3_flag",
        label: "Phase 3 refine",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-phase3"]
      },
      {
        id: "no_multishell_flag",
        label: "Multishell scoring",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-multishell"]
      },
      {
        id: "no_axis_pdb_flag",
        label: "Axis trace PDB",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-axis-pdb"]
      }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/symmetry_find", artifact_type: "dir" },
      {
        id: "symmetry_find_json",
        label: "Run summary JSON",
        default: "outputs/symmetry_find/symmetry_find.json",
        artifact_type: "table.json"
      },
      {
        id: "axis_pdb",
        label: "Axis CA trace PDB (if phase 4 runs)",
        default: "outputs/symmetry_find/symmetry_axis_ca.pdb",
        artifact_type: "model.structure"
      }
    ],
    arg_builders: {
      mask_arg: { when_input_present: "mask", value_template: '--mask "{mask}"' }
    },
    param_arg_builders: {
      density_threshold_arg: { when_param_present: "density_threshold", value_template: "--density-threshold {density_threshold}" },
      orders_arg: { when_param_present: "orders", value_template: '--orders "{orders}"' },
      guided_order_arg: { when_param_present: "guided_order", value_template: "--guided-order {guided_order}" },
      no_diagonals_flag: { when_param_present: "no_diagonals_flag", value_template: "{no_diagonals_flag}" },
      no_phase3_flag: { when_param_present: "no_phase3_flag", value_template: "{no_phase3_flag}" },
      no_multishell_flag: { when_param_present: "no_multishell_flag", value_template: "{no_multishell_flag}" },
      no_axis_pdb_flag: { when_param_present: "no_axis_pdb_flag", value_template: "{no_axis_pdb_flag}" }
    }
  },
  helical_find_run: {
    display_name: "Helical symmetry find",
    description:
      "Estimate **helical** rise and twist by maximizing one-step screw self-correlation (axis search: cardinal / PCA). " +
      "CLI: `cryomodel helical find`. Narrow twist/rise grids for slow-pitch fibrils.",
    command_template:
      'cryomodel helical find "{input_map}" "{out_dir}" --density-percentile {density_percentile} --axis-mode {axis_mode} --twist-min-deg {twist_min_deg} --twist-max-deg {twist_max_deg} --twist-step-deg {twist_step_deg} --rise-min-A {rise_min_A} --rise-max-A {rise_max_A} --rise-step-A {rise_step_A} --max-voxels-score {max_voxels_score} --seed {seed} --refine-iters {refine_iters} {density_threshold_arg} {no_refine_flag} {no_heatmap_flag}',
    inputs: [{ id: "input_map", label: "Input map (.mrc)", required: true, artifact_type: "map.mrc" }],
    params: [
      {
        id: "density_percentile",
        label: "Density percentile (voxel mask)",
        type: "float",
        required: true,
        default: 90.0,
        min: 0.0,
        max: 100.0
      },
      {
        id: "density_threshold",
        label: "Absolute density threshold (optional; overrides percentile when set)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "axis_mode",
        label: "Axis search mode",
        type: "string",
        required: true,
        default: "cardinal_pca",
        options: ["cardinal", "pca", "cardinal_pca"]
      },
      { id: "twist_min_deg", label: "Twist min (deg)", type: "float", required: true, default: -20.0, min: -180.0, max: 180.0 },
      { id: "twist_max_deg", label: "Twist max (deg)", type: "float", required: true, default: 20.0, min: -180.0, max: 180.0 },
      { id: "twist_step_deg", label: "Twist step (deg)", type: "float", required: true, default: 0.5, min: 0.05, max: 90.0 },
      { id: "rise_min_A", label: "Rise min (Å)", type: "float", required: true, default: 2.0, min: 0.1, max: 500.0 },
      { id: "rise_max_A", label: "Rise max (Å)", type: "float", required: true, default: 8.0, min: 0.1, max: 500.0 },
      { id: "rise_step_A", label: "Rise step (Å)", type: "float", required: true, default: 0.2, min: 0.02, max: 100.0 },
      {
        id: "max_voxels_score",
        label: "Max voxels for scoring",
        type: "int",
        required: true,
        default: 200000,
        min: 5000,
        max: 5_000_000
      },
      { id: "seed", label: "Random seed", type: "int", required: true, default: 0, min: 0, max: 2_147_483_647 },
      {
        id: "refine_iters",
        label: "Refine iterations (after coarse best; 0 = coarse only)",
        type: "int",
        required: true,
        default: 2,
        min: 0,
        max: 8
      },
      {
        id: "no_refine_flag",
        label: "Local refinement",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-refine"]
      },
      {
        id: "no_heatmap_flag",
        label: "Score heatmap PNG",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-heatmap"]
      }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/helical_find", artifact_type: "dir" },
      {
        id: "helical_find_json",
        label: "Helical parameters JSON",
        default: "outputs/helical_find/helical_find.json",
        artifact_type: "table.json"
      },
      {
        id: "heatmap_png",
        label: "Twist vs rise heatmap (unless skipped)",
        default: "outputs/helical_find/helical_score_heatmap.png",
        artifact_type: ""
      }
    ],
    param_arg_builders: {
      density_threshold_arg: { when_param_present: "density_threshold", value_template: "--density-threshold {density_threshold}" },
      no_refine_flag: { when_param_present: "no_refine_flag", value_template: "{no_refine_flag}" },
      no_heatmap_flag: { when_param_present: "no_heatmap_flag", value_template: "{no_heatmap_flag}" }
    }
  },
  helical_segment_run: {
    display_name: "Helical segment",
    description:
      "Label map voxels into helical subunits from **helical find** JSON (axis, rise, twist). " +
      "CLI: `cryomodel helical segment`. Inherit **helical_find_json** from a Helical symmetry find card when possible.",
    command_template:
      'cryomodel helical segment "{input_map}" "{helical_json}" "{out_dir}" --k-window {k_window} --max-norm-cost {max_norm_cost} --min-cost-margin {min_cost_margin} --mode {mode} --radial-band-halfwidth-A {radial_band_halfwidth_A} --peak-min-prominence {peak_min_prominence} {density_threshold_arg} {sigma_t_arg} {sigma_phi_arg} {radial_band_center_arg} {axial_window_arg} {shear_alpha_arg} {shear_pos_arg} {shear_neg_arg} {watershed_max_cost_arg} {largest_component_flag} {prune_labels_flag} {no_qc_png_flag} {no_average_flag} {no_sequential_labels_flag}',
    inputs: [
      { id: "input_map", label: "Input map (.mrc)", required: true, artifact_type: "map.mrc" },
      {
        id: "helical_json",
        label: "Helical find JSON (helical_find.json)",
        required: true,
        artifact_type: "table.json"
      }
    ],
    params: [
      {
        id: "k_window",
        label: "Local k search half-window",
        type: "int",
        required: true,
        default: 3,
        min: 1,
        max: 12
      },
      {
        id: "max_norm_cost",
        label: "Max normalized assignment cost",
        type: "float",
        required: true,
        default: 12.0,
        min: 0.1,
        max: 100.0
      },
      {
        id: "min_cost_margin",
        label: "Min gap best vs 2nd-best assignment",
        type: "float",
        required: true,
        default: 0.05,
        min: 0.0,
        max: 50.0
      },
      {
        id: "mode",
        label: "Segmentation mode",
        type: "string",
        required: true,
        default: "phase_peaks",
        options: ["phase_peaks", "seeded_watershed", "analytic"]
      },
      {
        id: "radial_band_halfwidth_A",
        label: "Radial band halfwidth (Å)",
        type: "float",
        required: true,
        default: 2.5,
        min: 0.3,
        max: 30.0
      },
      {
        id: "peak_min_prominence",
        label: "Min peak prominence (axial profile)",
        type: "float",
        required: true,
        default: 0.0,
        min: 0.0,
        max: 1e6
      },
      {
        id: "density_threshold",
        label: "Density threshold (optional; default from helical JSON)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "sigma_t_A",
        label: "Axial tolerance σ_t (Å, optional)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "sigma_phi_deg",
        label: "Angular tolerance σ_φ (deg, optional)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "radial_band_center_A",
        label: "Radial band center (Å, optional)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "axial_window_halfwidth_A",
        label: "Repeat window halfwidth (Å, optional)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "shear_alpha_rad_per_A",
        label: "Shear α (rad/Å, optional; single-slope)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "shear_alpha_pos_rad_per_A",
        label: "Shear α+ for Δz≥0 (optional; with α−)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "shear_alpha_neg_rad_per_A",
        label: "Shear α− for Δz<0 (optional; with α+)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "watershed_max_norm_cost",
        label: "Watershed fill cost cap (optional)",
        type: "string",
        required: false,
        default: ""
      },
      {
        id: "largest_component_flag",
        label: "Representative: largest 26-connected component only",
        type: "string",
        required: false,
        default: "",
        options: ["", "--largest-component"]
      },
      {
        id: "prune_labels_flag",
        label: "Labels: per-ID largest component only",
        type: "string",
        required: false,
        default: "",
        options: ["", "--prune-labels-largest-component"]
      },
      {
        id: "no_qc_png_flag",
        label: "QC diagnostic PNG",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-qc-png"]
      },
      {
        id: "no_average_flag",
        label: "Average-subunit map",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-average"]
      },
      {
        id: "no_sequential_labels_flag",
        label: "Subunit IDs",
        type: "string",
        required: false,
        default: "",
        options: ["", "--no-sequential-helical-labels"]
      }
    ],
    outputs: [
      { id: "out_dir", label: "Output directory", default: "outputs/helical_segment", artifact_type: "dir" },
      {
        id: "labels_mrc",
        label: "Subunit label map",
        default: "outputs/helical_segment/helical_subunit_labels.mrc",
        artifact_type: "map.mrc"
      },
      {
        id: "representative_mrc",
        label: "Representative subunit map",
        default: "outputs/helical_segment/helical_subunit_representative.mrc",
        artifact_type: "map.mrc"
      },
      {
        id: "average_mrc",
        label: "Average subunit map (if not skipped)",
        default: "outputs/helical_segment/helical_subunit_average.mrc",
        artifact_type: "map.mrc"
      },
      {
        id: "segment_json",
        label: "Segmentation summary JSON",
        default: "outputs/helical_segment/helical_segment.json",
        artifact_type: "table.json"
      },
      {
        id: "qc_png",
        label: "QC PNG (unless skipped)",
        default: "outputs/helical_segment/helical_segment_qc.png",
        artifact_type: ""
      }
    ],
    param_arg_builders: {
      density_threshold_arg: { when_param_present: "density_threshold", value_template: "--density-threshold {density_threshold}" },
      sigma_t_arg: { when_param_present: "sigma_t_A", value_template: "--sigma-t-A {sigma_t_A}" },
      sigma_phi_arg: { when_param_present: "sigma_phi_deg", value_template: "--sigma-phi-deg {sigma_phi_deg}" },
      radial_band_center_arg: { when_param_present: "radial_band_center_A", value_template: "--radial-band-center-A {radial_band_center_A}" },
      axial_window_arg: { when_param_present: "axial_window_halfwidth_A", value_template: "--axial-window-halfwidth-A {axial_window_halfwidth_A}" },
      shear_alpha_arg: { when_param_present: "shear_alpha_rad_per_A", value_template: "--shear-alpha-rad-per-A {shear_alpha_rad_per_A}" },
      shear_pos_arg: { when_param_present: "shear_alpha_pos_rad_per_A", value_template: "--shear-alpha-pos-rad-per-A {shear_alpha_pos_rad_per_A}" },
      shear_neg_arg: { when_param_present: "shear_alpha_neg_rad_per_A", value_template: "--shear-alpha-neg-rad-per-A {shear_alpha_neg_rad_per_A}" },
      watershed_max_cost_arg: { when_param_present: "watershed_max_norm_cost", value_template: "--watershed-max-norm-cost {watershed_max_norm_cost}" },
      largest_component_flag: { when_param_present: "largest_component_flag", value_template: "{largest_component_flag}" },
      prune_labels_flag: { when_param_present: "prune_labels_flag", value_template: "{prune_labels_flag}" },
      no_qc_png_flag: { when_param_present: "no_qc_png_flag", value_template: "{no_qc_png_flag}" },
      no_average_flag: { when_param_present: "no_average_flag", value_template: "{no_average_flag}" },
      no_sequential_labels_flag: { when_param_present: "no_sequential_labels_flag", value_template: "{no_sequential_labels_flag}" }
    }
  }
};

/** Per tool type: production | testing | untested | experimental | broken — adjust as you verify tools. */
const SPECS_DEV_STATUS = {
  mapfilter_apply: "production",
  model2map_convert: "production",
  affilter_run: "production",
  dnaaxis_extract: "production",
  pathmeasure_launcher: "production",
  foldhunter_search: "testing",
  findligands_run: "untested",
  predictligands_run: "untested",
  fitprep_check: "untested",
  validate_run: "untested",
  pathwalker2_discover: "untested",
  pathwalker_run: "untested",
  pathwalker_average_run: "untested",
  pyhole_analyze: "untested",
  pyhole_plot_run: "untested",
  train_ml_run: "experimental",
  train_ensemble_run: "experimental",
  extract_features_run: "experimental",
  loopcloud_generate: "untested",
  fitcompare_run: "untested",
  pdbdomain_identify: "untested",
  pdbcom_compute: "untested",
  dnabuild_build: "production",
  basehunter_run: "untested",
  pdb_mutate_run: "testing",
  zonal_refine_run: "testing",
  zonal_refine_global_run: "testing",
  alignment_sequence_pick_run: "testing",
  seqconservation_run: "testing",
  seqconservation_diffuse_run: "experimental",
  symmetry_find_run: "testing",
  helical_find_run: "testing",
  helical_segment_run: "testing"
};

const DEV_STATUS_LABELS = {
  production: "Production",
  testing: "In testing",
  untested: "Untested",
  experimental: "Experimental",
  broken: "Broken"
};

for (const k of Object.keys(SPECS)) {
  SPECS[k].dev_status = SPECS_DEV_STATUS[k] || "testing";
}
  global.SPECS = SPECS;
  global.SPECS_DEV_STATUS = SPECS_DEV_STATUS;
  global.DEV_STATUS_LABELS = DEV_STATUS_LABELS;
})(typeof globalThis !== "undefined" ? globalThis : this);
