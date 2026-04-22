/**
 * Maps workflow card job_type → CryoModel workflow doc tool/command (shared with import/export).
 */
(function (global) {
  const JOB_TOOL_COMMAND = {
    mapfilter_apply: { tool: "mapfilter", command: "apply" },
    model2map_convert: { tool: "model2map", command: "" },
    affilter_run: { tool: "affilter", command: "" },
    foldhunter_search: { tool: "foldhunter", command: "" },
    findligands_run: { tool: "findligands", command: "" },
    predictligands_run: { tool: "predictligands", command: "" },
    fitprep_check: { tool: "fitprep", command: "" },
    validate_run: { tool: "validate", command: "" },
    pathwalker2_discover: { tool: "pathwalker2", command: "" },
    pathwalker_run: { tool: "pathwalker", command: "" },
    pathwalker_average_run: { tool: "pathwalker-average", command: "" },
    pyhole_analyze: { tool: "pyhole", command: "analyze" },
    pyhole_plot_run: { tool: "pyhole-plot", command: "plot" },
    train_ml_run: { tool: "train-ml", command: "" },
    train_ensemble_run: { tool: "train-ensemble", command: "" },
    extract_features_run: { tool: "extract-features", command: "" },
    loopcloud_generate: { tool: "loopcloud", command: "" },
    fitcompare_run: { tool: "fitcompare", command: "" },
    pdbdomain_identify: { tool: "pdbdomain", command: "" },
    pdbcom_compute: { tool: "pdbcom", command: "" },
    dnaaxis_extract: { tool: "dnaaxis", command: "extract" },
    dnabuild_build: { tool: "dnabuild", command: "build-2bp" },
    basehunter_run: { tool: "basehunter", command: "compare" },
    pdb_mutate_run: { tool: "pdb-mutate", command: "run" },
    zonal_refine_run: { tool: "zonal-refine", command: "run" },
    zonal_refine_global_run: { tool: "zonal-refine", command: "global" },
    alignment_sequence_pick_run: { tool: "fasta-extract", command: "row" },
    seqconservation_run: { tool: "seqconservation", command: "" },
    seqconservation_diffuse_run: { tool: "seqconservation-diffuse", command: "" },
    symmetry_find_run: { tool: "symmetry", command: "find" },
    helical_find_run: { tool: "helical", command: "find" },
    helical_segment_run: { tool: "helical", command: "segment" }
  };
  global.JOB_TOOL_COMMAND = JOB_TOOL_COMMAND;
})(typeof globalThis !== "undefined" ? globalThis : this);
