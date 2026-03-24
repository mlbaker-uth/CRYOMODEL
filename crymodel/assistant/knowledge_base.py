# crymodel/assistant/knowledge_base.py
"""Structured knowledge base for CryoModel AI assistant."""
from typing import Dict, List, Any

# Tool descriptions with purpose, inputs, outputs, common use cases, and error patterns
TOOL_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "findligands": {
        "purpose": "Identify and classify unmodeled density as waters/ions and ligands",
        "description": "Masks the model from the map, separates water-sized blobs from ligand-sized blobs, generates pseudoatoms, and optionally uses ML to classify waters/ions.",
        "inputs": {
            "required": ["map", "model"],
            "optional": ["threshold", "mask_radius", "half1", "half2", "ml_model", "entry_resolution"]
        },
        "outputs": {
            "waters_map.mrc": "Density map containing only water-sized blobs",
            "ligands_map.mrc": "Density map containing ligand-sized blobs",
            "water.pdb": "Pseudoatoms for water candidates",
            "ligands.pdb": "Pseudoatoms for ligand candidates",
            "sites.csv": "Feature table for all identified sites",
            "water-predictions.csv": "ML predictions for water/ion classification (if ML enabled)"
        },
        "common_use_cases": [
            "Finding missing waters in high-resolution maps (≤2.5 Å)",
            "Identifying ligand binding sites",
            "Preparing data for ML classification",
            "Analyzing unmodeled density"
        ],
        "typical_workflow": "findligands → predictligands (for ligand classification)",
        "parameter_guidance": {
            "thresh": "Density threshold (typically 0.3-0.7). Lower = more sensitive, higher = more selective.",
            "mask_radius": "Radius around model atoms to mask (default 2.0 Å). Increase for flexible regions.",
            "micro_vvox_min": "Minimum voxels for water/ion blobs (default 2). Adjust based on map resolution.",
            "micro_vvox_max": "Maximum voxels for water/ion blobs (default 12). Adjust based on map resolution.",
            "zero_radius": "Zero radius for greedy water picking (default 2.0 Å).",
            "entry_resolution": "Map resolution for ML filtering. Critical for accurate water/ion classification."
        },
        "common_errors": {
            "empty_pdb_files": {
                "symptoms": ["Empty water.pdb or ligands.pdb files", "No pseudoatoms generated"],
                "solutions": [
                    "Check threshold - may be too high (try lowering to 0.3-0.4)",
                    "Verify map/model alignment with fitprep",
                    "Check that map contains unmodeled density",
                    "Try adjusting mask_radius (may be masking too much)"
                ]
            },
            "no_ligands_found": {
                "symptoms": ["ligands_map.mrc is empty", "No ligand pseudoatoms"],
                "solutions": [
                    "Lower threshold to detect smaller blobs",
                    "Check if ligands are already modeled in PDB",
                    "Verify map quality in ligand region",
                    "Try different micro_vvox_max (may be filtering out ligands)"
                ]
            },
            "ml_prediction_failed": {
                "symptoms": ["ML model not found", "Feature mismatch errors"],
                "solutions": [
                    "Ensure --ml-model points to trained model directory",
                    "Check model was trained with compatible feature set",
                    "Verify --entry-resolution matches map resolution"
                ]
            }
        }
    },
    
    "foldhunter": {
        "purpose": "Fit AlphaFold models to cryo-EM density maps using FFT-based cross-correlation",
        "description": "Exhaustive search for optimal fit of probe structure (AlphaFold model) to target density map using coarse-to-fine refinement.",
        "inputs": {
            "required": ["target_map", "probe_pdb or probe_map"],
            "optional": ["resolution", "plddt_threshold", "density_threshold", "symmetry", "downsample_factor"]
        },
        "outputs": {
            "foldhunter_top_fit.pdb": "Best-fit transformed probe structure",
            "foldhunter_results.csv": "Ranked list of candidate fits with scores"
        },
        "common_use_cases": [
            "Initial model placement in density",
            "Domain fitting",
            "Template-based modeling",
            "Validating AlphaFold models against experimental data"
        ],
        "typical_workflow": "affilter → foldhunter → loopcloud (rebuild low pLDDT regions)",
        "parameter_guidance": {
            "resolution": "Should match target map resolution (typically 2.5-4.0 Å). Used for probe map generation.",
            "plddt_threshold": "Minimum pLDDT to include atoms (default 0.5). Lower (0.3) for more atoms, higher (0.7) for high-confidence only.",
            "n_coarse_rotations": "Number of rotations for coarse search (default 1000). Increase for better coverage.",
            "n_fine_rotations": "Number of rotations for fine refinement (default 500). Increase for better accuracy.",
            "symmetry": "Rotational symmetry (e.g., 3 for C3). Reduces search space.",
            "downsample_factor": "Downsampling for coarse search (None = auto). Use 2 for maps >256 voxels."
        },
        "common_errors": {
            "low_correlation": {
                "symptoms": ["Correlation < 0.3", "Poor fit quality"],
                "solutions": [
                    "Check map resolution matches probe resolution parameter",
                    "Try different pLDDT threshold (0.3-0.7)",
                    "Verify map/model alignment with fitprep",
                    "Check if probe structure matches map (wrong domain?)",
                    "Increase n_coarse_rotations for better search coverage"
                ]
            },
            "out_of_memory": {
                "symptoms": ["Memory errors", "Process killed"],
                "solutions": [
                    "Use --downsample-factor 2 for large maps",
                    "Reduce n_coarse_rotations",
                    "Use smaller probe structure (filter more with affilter)"
                ]
            },
            "no_atoms_in_density": {
                "symptoms": ["Atom inclusion score = 0", "All atoms outside map"],
                "solutions": [
                    "Check map boundaries - probe may be outside",
                    "Verify map origin and apix are correct",
                    "Check if probe needs to be centered differently"
                ]
            }
        }
    },
    
    "affilter": {
        "purpose": "Filter AlphaFold models to remove low-quality regions and identify domains",
        "description": "Removes low pLDDT regions, extended loops (barbed wire), and low-connectivity artifacts. Identifies structural domains using clustering.",
        "inputs": {
            "required": ["input_pdb"],
            "optional": ["plddt_threshold", "filter_loops", "filter_connectivity", "clustering_method"]
        },
        "outputs": {
            "alphafold_filtered.pdb": "Filtered structure with low-quality regions removed",
            "affilter_low_plddt_regions.csv": "List of low pLDDT regions (for loop modeling)",
            "affilter_domains.csv": "Identified structural domains",
            "affilter_stats.txt": "Filtering statistics"
        },
        "common_use_cases": [
            "Preparing AlphaFold models for fitting",
            "Identifying domains for domain-based fitting",
            "Finding regions to rebuild with loop modeling",
            "Quality assessment of AlphaFold predictions"
        ],
        "typical_workflow": "affilter → foldhunter → loopcloud (rebuild filtered regions)",
        "parameter_guidance": {
            "plddt_threshold": "Minimum pLDDT to keep (default 0.5). Lower = keep more, higher = more selective.",
            "filter_loops": "Remove extended loops (default True). Disable if loops are real.",
            "filter_connectivity": "Remove low-connectivity artifacts (default True). Disable if structure is fragmented.",
            "clustering_method": "Domain identification method: 'dbscan' (default) or 'agglomerative'."
        },
        "common_errors": {
            "too_much_filtered": {
                "symptoms": ["Most of structure removed", "Very few residues remaining"],
                "solutions": [
                    "Lower plddt_threshold (try 0.3-0.4)",
                    "Disable filter_loops if loops are real",
                    "Disable filter_connectivity if structure is fragmented",
                    "Check if input model has very low pLDDT overall"
                ]
            },
            "no_domains_found": {
                "symptoms": ["No domains identified", "Empty domains CSV"],
                "solutions": [
                    "Adjust clustering_eps (default 15.0 Å) - increase for larger domains",
                    "Reduce clustering_min_samples for smaller domains",
                    "Try agglomerative clustering with explicit n_clusters"
                ]
            }
        }
    },
    
    "pathwalker": {
        "purpose": "Trace protein backbone through density map using TSP solver",
        "description": "Generates pseudoatoms equal to C-alpha count, solves TSP to find optimal backbone path.",
        "inputs": {
            "required": ["map", "model"],
            "optional": ["threshold", "n_iterations", "solver"]
        },
        "outputs": {
            "path.pdb": "Traced backbone path",
            "path_statistics.json": "Path quality metrics"
        },
        "common_use_cases": [
            "De novo backbone tracing",
            "Validating existing models",
            "Finding alternative conformations"
        ],
        "typical_workflow": "pathwalker → validate (check path quality)",
        "parameter_guidance": {
            "threshold": "Density threshold for pseudoatom seeding (typically 0.05-0.2). Lower = more sensitive.",
            "n_iterations": "Number of optimization iterations (default 5). More = better but slower."
        },
        "common_errors": {
            "too_few_pseudoatoms": {
                "symptoms": ["Fewer pseudoatoms than C-alphas", "Incomplete path"],
                "solutions": [
                    "Lower threshold to detect more density",
                    "Check map quality and resolution",
                    "Verify map/model alignment"
                ]
            },
            "poor_path_quality": {
                "symptoms": ["High C-alpha distance errors", "Discontinuous path"],
                "solutions": [
                    "Increase n_iterations for more optimization",
                    "Try different threshold",
                    "Check map resolution (may be too low)",
                    "Use pathwalker2 for more robust tracing"
                ]
            }
        }
    },
    
    "pathwalker2": {
        "purpose": "Modern backbone tracing with advanced pseudoatom detection and protein geometry constraints",
        "description": "Step 1 of Pathwalker 2.0: Automatic trace discovery with ridge detection, helix detection, and VRP routing.",
        "inputs": {
            "required": ["map", "threshold", "n_residues_estimate"],
            "optional": ["seeding_method", "ss_helix", "routing_k", "weights"]
        },
        "outputs": {
            "pathwalker2_fragments.pdb": "Ranked C-alpha fragments",
            "pathwalker2_meta.json": "Tracing metadata and statistics"
        },
        "common_use_cases": [
            "De novo backbone tracing at medium resolution",
            "Domain tracing",
            "Alternative conformation detection"
        ],
        "typical_workflow": "pathwalker2 discover → pathwalker2 thread (future)",
        "parameter_guidance": {
            "threshold": "Density threshold (typically 0.01-0.05). Critical for pseudoatom seeding.",
            "n_residues_estimate": "Estimated number of residues. Should match actual protein size.",
            "map_prep": "Map preprocessing: 'gaussian', 'locscale', or 'none'. 'locscale' recommended for noisy maps."
        },
        "common_errors": {
            "too_few_pseudoatoms": {
                "symptoms": ["Much fewer pseudoatoms than n_residues", "Incomplete coverage"],
                "solutions": [
                    "Lower threshold (try 0.01-0.02)",
                    "Check map quality - may need better preprocessing",
                    "Verify n_residues_estimate is correct",
                    "Try different map_prep method"
                ]
            },
            "pseudoatoms_in_sidechains": {
                "symptoms": ["Pseudoatoms clustered in side-chain density", "Not on backbone"],
                "solutions": [
                    "This is a known issue - code uses thickness penalty to avoid side chains",
                    "Check if map resolution is sufficient for backbone detection",
                    "Try adjusting weight parameters (w_t for thickness penalty)"
                ]
            }
        }
    },
    
    "predictligands": {
        "purpose": "Classify ligand density blobs into known ligand classes",
        "description": "Uses rule-based classification to identify hemes, nucleotides, phospholipids, NAD, and ubiquinone.",
        "inputs": {
            "required": ["ligands_pdb", "ligand_map", "model"],
            "optional": ["min_pseudoatoms", "max_pseudoatoms", "threshold_lower"]
        },
        "outputs": {
            "ligand-predictions.csv": "Classified ligands with confidence scores"
        },
        "common_use_cases": [
            "Identifying cofactors",
            "Characterizing ligand binding sites",
            "Preparing for refinement"
        ],
        "typical_workflow": "findligands → predictligands",
        "parameter_guidance": {
            "min_pseudoatoms": "Minimum pseudoatoms per component (default 3). Filters noise.",
            "max_pseudoatoms": "Maximum pseudoatoms per component (default 80). Prevents over-grouping."
        },
        "common_errors": {
            "no_predictions": {
                "symptoms": ["Empty predictions CSV", "All components filtered"],
                "solutions": [
                    "Lower min_pseudoatoms (try 1-2)",
                    "Check if ligands.pdb has pseudoatoms",
                    "Verify ligand_map.mrc contains density"
                ]
            },
            "over_grouping": {
                "symptoms": ["Very large components (100+ pseudoatoms)", "Multiple ligands grouped"],
                "solutions": [
                    "Lower max_pseudoatoms (try 50-60)",
                    "Adjust threshold_lower for grouping",
                    "Check distance_threshold parameter"
                ]
            }
        }
    },
    
    "basehunter": {
        "purpose": "Score and classify nucleic-acid bases in cryo-EM maps",
        "description": "Evaluates local density and geometry for each nucleotide, assigns a BaseHunter score, and classifies bases (purine/pyrimidine) to flag weak or inconsistent residues.",
        "inputs": {
            "required": ["map", "model"],
            "optional": ["chain", "resolution", "min_residues", "out_dir"]
        },
        "outputs": {
            "basehunter_scores.csv": "Per-residue scores and predicted base class with confidence",
            "basehunter_summary.json": "Global statistics and recommended thresholds"
        },
        "common_use_cases": [
            "Finding poorly supported bases in high-resolution DNA/RNA maps",
            "Checking purine/pyrimidine assignments against the map",
            "Prioritizing bases for manual rebuilding or sequence correction"
        ],
        "typical_workflow": "dnaaxis → dnabuild → basehunter (for DNA path + base QC)",
        "parameter_guidance": {
            "resolution": "Map resolution in Å (e.g. 2.5–3.5). Used to tune density features; set to the refinement resolution.",
            "chain": "Restrict analysis to a specific nucleic-acid chain (e.g. A). Omit to analyze all chains.",
            "min_residues": "Minimum contiguous nucleotides per segment to score (default 4). Lower for very short fragments."
        }
    },
    
    "validate": {
        "purpose": "Assess cryoEM models with resolution-dependent validation metrics",
        "description": "Computes Ringer-lite, Q-lite, Cα-tube, local CC, and geometry features. Provides per-residue quality scores.",
        "inputs": {
            "required": ["model", "map"],
            "optional": ["half1", "half2", "localres", "priors_file"]
        },
        "outputs": {
            "fitcheck_per_residue.csv": "Per-residue validation metrics",
            "fitcheck_summary.txt": "Global quality summary"
        },
        "common_use_cases": [
            "Model quality assessment",
            "Identifying problematic regions",
            "Resolution-dependent validation"
        ],
        "typical_workflow": "Any modeling tool → validate",
        "parameter_guidance": {
            "half1/half2": "Half-maps for FSC-based validation. Highly recommended if available.",
            "localres": "Local resolution map. Improves resolution-aware scoring."
        }
    }
}

# Workflow templates for common tasks
WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "build_model_from_sequence": {
        "description": "Build a complete model from protein sequence and density map",
        "prerequisites": ["sequence", "density_map", "alphafold_model"],
        "steps": [
            {
                "step": 1,
                "tool": "affilter",
                "action": "Filter AlphaFold model to remove low pLDDT regions",
                "command": "crymodel affilter filter alphafold_model.pdb --plddt-threshold 0.5 --out-dir outputs/affilter"
            },
            {
                "step": 2,
                "tool": "foldhunter",
                "action": "Fit filtered model to density map",
                "command": "crymodel foldhunter search target_map.mrc --probe-pdb outputs/affilter/alphafold_filtered.pdb --resolution 3.0 --out-dir outputs/foldhunter"
            },
            {
                "step": 3,
                "tool": "loopcloud",
                "action": "Rebuild low pLDDT regions identified by affilter",
                "command": "crymodel loopcloud generate outputs/foldhunter/foldhunter_top_fit.pdb 'A:100 -> A:120' 'GGSGG' --map target_map.mrc --out-dir outputs/loopcloud"
            },
            {
                "step": 4,
                "tool": "validate",
                "action": "Validate final model",
                "command": "crymodel validate outputs/loopcloud/best_loop.pdb target_map.mrc --half1 half1.mrc --half2 half2.mrc --out-dir outputs/validate"
            }
        ],
        "notes": [
            "Step 3 requires manual identification of loop regions from affilter_low_plddt_regions.csv",
            "For automated workflow, use workflow.yaml file"
        ]
    },
    
    "find_and_classify_ligands": {
        "description": "Find unmodeled density and classify as waters/ions/ligands",
        "prerequisites": ["density_map", "model_pdb"],
        "steps": [
            {
                "step": 1,
                "tool": "findligands",
                "action": "Identify water and ligand density",
                "command": "crymodel findligands --map target_map.mrc --model model.pdb --thresh 0.5 --ml-model trained_model --entry-resolution 2.8 --out-dir outputs/findligands"
            },
            {
                "step": 2,
                "tool": "predictligands",
                "action": "Classify ligand components",
                "command": "crymodel predictligands --ligands-pdb outputs/findligands/ligands.pdb --ligand-map outputs/findligands/ligands_map.mrc --model model.pdb --out-dir outputs/predictligands"
            }
        ],
        "notes": [
            "ML model required for water/ion classification in step 1",
            "Resolution parameter critical for accurate classification"
        ]
    },
    "ligand_qc": {
        "description": "Find ligands and validate model in density",
        "prerequisites": ["density_map", "model_pdb"],
        "steps": [
            {
                "step": 1,
                "tool": "findligands",
                "action": "Identify water and ligand density",
                "command": "crymodel findligands --map target_map.mrc --model model.pdb --thresh 0.5 --ml-model trained_model --entry-resolution 2.8 --out-dir outputs/findligands"
            },
            {
                "step": 2,
                "tool": "predictligands",
                "action": "Classify ligand components",
                "command": "crymodel predictligands --ligands-pdb outputs/findligands/ligands.pdb --ligand-map outputs/findligands/ligands_map.mrc --model model.pdb --out-dir outputs/predictligands"
            },
            {
                "step": 3,
                "tool": "validate",
                "action": "Validate model against map",
                "command": "crymodel validate --model model.pdb --map target_map.mrc --out-dir outputs/validate"
            }
        ],
        "notes": [
            "Validation helps interpret ligand confidence in context of model quality"
        ]
    },
    
    "trace_backbone": {
        "description": "Trace protein backbone through density map",
        "prerequisites": ["density_map"],
        "steps": [
            {
                "step": 1,
                "tool": "pathwalker2",
                "action": "Discover backbone traces",
                "command": "crymodel pathwalker2 discover target_map.mrc --threshold 0.013 --n-residues 172 --out-dir outputs/pathwalker2"
            },
            {
                "step": 2,
                "tool": "validate",
                "action": "Assess trace quality",
                "command": "crymodel validate outputs/pathwalker2/pathwalker2_fragments.pdb target_map.mrc --out-dir outputs/validate"
            }
        ],
        "notes": [
            "n_residues should match expected protein size",
            "Threshold may need adjustment based on map quality"
        ]
    },
    "trace_dna_centerline": {
        "description": "Trace dsDNA centerline and build a poly-AT model",
        "prerequisites": ["density_map"],
        "steps": [
            {
                "step": 1,
                "tool": "dnaaxis",
                "action": "Trace centerline through dsDNA density",
                "command": "crymodel dnaaxis extract --map dna_map.mrc --threshold 0.25 --guides-pdb guides.pdb --out-pdb dna_axis.pdb --out-mrc dna_axis.mrc"
            },
            {
                "step": 2,
                "tool": "dnabuild",
                "action": "Build poly-AT model from centerline",
                "command": "crymodel dnabuild build-2bp --centerline-pdb dna_axis.pdb --template-2bp-pdb data/DNA-TEMPLATES/2AT-template.pdb --out-pdb dna_model.pdb"
            }
        ],
        "notes": [
            "Guides are recommended for curved DNA or noisy maps",
            "Adjust threshold based on map filtering and blur"
        ]
    },
    "dna_basehunter_build": {
        "description": "Trace DNA axis, build a model, then classify bases with BaseHunter",
        "prerequisites": ["density_map"],
        "steps": [
            {
                "step": 1,
                "tool": "dnaaxis",
                "action": "Trace centerline through dsDNA density",
                "command": "crymodel dnaaxis extract --map dna_map.mrc --threshold 0.25 --guides-pdb guides.pdb --out-pdb dna_axis.pdb --out-mrc dna_axis.mrc"
            },
            {
                "step": 2,
                "tool": "dnabuild",
                "action": "Build a DNA model from the extracted axis",
                "command": "crymodel dnabuild build --centerline-pdb dna_axis.pdb --map dna_map.mrc --out-pdb dna_initial.pdb --resolution 3.0"
            },
            {
                "step": 3,
                "tool": "basehunter",
                "action": "Classify bases and generate a summary table",
                "command": "crymodel basehunter --map dna_map.mrc --model dna_initial.pdb --out-dir outputs/basehunter --resolution 3.0"
            }
        ],
        "notes": [
            "Use BaseHunter scores and predicted classes to flag weak or inconsistent bases",
            "Adjust thresholds and resolution based on map quality"
        ]
    },
    "basehunter_classify": {
        "description": "Classify purine/pyrimidine base pairs with BaseHunter",
        "prerequisites": ["density_map", "base_pair_list"],
        "steps": [
            {
                "step": 1,
                "tool": "basehunter",
                "action": "Compare base-pair densities against templates",
                "command": "crymodel basehunter compare --input-file base_pairs.txt --threshold 0.45 --out-dir outputs/basehunter"
            }
        ],
        "notes": [
            "Use per-pair thresholds in input file if map is heterogeneous",
            "Template thresholds may differ from input thresholds"
        ]
    }
}

# Common error patterns and solutions
ERROR_PATTERNS: Dict[str, Dict[str, Any]] = {
    "ImportError": {
        "patterns": ["cannot import", "ImportError", "ModuleNotFoundError"],
        "solutions": [
            "Check virtual environment is activated: source .venv/bin/activate",
            "Reinstall package: pip install -e .",
            "Check dependencies in pyproject.toml",
            "Try: pip install --upgrade -r requirements.txt"
        ],
        "prevention": "Always work within virtual environment"
    },
    
    "empty_output": {
        "patterns": ["no results", "empty file", "0 pseudoatoms", "no candidates"],
        "solutions": [
            "Check input threshold - may be too restrictive (try lowering)",
            "Verify map/model alignment with fitprep",
            "Check map quality and resolution",
            "Ensure input files are correct format and not corrupted"
        ],
        "prevention": "Always validate inputs before running tools"
    },
    
    "memory_error": {
        "patterns": ["MemoryError", "out of memory", "killed", "OOM"],
        "solutions": [
            "Use downsampling for large maps (--downsample-factor 2)",
            "Reduce search space (fewer rotations, smaller maps)",
            "Process maps in chunks if possible",
            "Increase system memory or use smaller inputs"
        ],
        "prevention": "Check map size before processing large datasets"
    },
    
    "file_not_found": {
        "patterns": ["FileNotFoundError", "No such file", "cannot open"],
        "solutions": [
            "Check file paths are correct (absolute or relative to current directory)",
            "Verify files exist: ls -la <file_path>",
            "Check file permissions",
            "Ensure file extensions are correct (.mrc, .pdb, etc.)"
        ],
        "prevention": "Always use absolute paths or verify relative paths"
    },
    "permission_denied": {
        "patterns": ["Permission denied", "permission error", "EACCES"],
        "solutions": [
            "Check file permissions and ownership",
            "Write outputs to a directory you own",
            "Avoid writing to system-protected folders"
        ],
        "prevention": "Use project-local output directories"
    },
    "invalid_mrc": {
        "patterns": ["bad mrc", "CCP4 map", "unsupported map", "not a CCP4", "mrc header"],
        "solutions": [
            "Verify the map opens in ChimeraX/Phenix",
            "Re-export or re-save map as MRC/CCP4",
            "Check for corrupted or truncated files"
        ],
        "prevention": "Use standard MRC/CCP4 writers"
    },
    "mismatched_box": {
        "patterns": ["shape mismatch", "different box", "incompatible dimensions"],
        "solutions": [
            "Resample maps to the same grid and voxel size",
            "Check that map and templates share consistent box size",
            "Verify apix/origin match before comparison"
        ],
        "prevention": "Normalize grids during preprocessing"
    },
    "threshold_too_high": {
        "patterns": ["empty mask", "no voxels above threshold", "zero density after threshold"],
        "solutions": [
            "Lower threshold and re-run",
            "Use histogram or map stats to pick a threshold",
            "If map was low-pass filtered, re-scale intensities"
        ],
        "prevention": "Inspect map stats before choosing thresholds"
    },
    
    "coordinate_mismatch": {
        "patterns": ["coordinate", "origin", "apix", "alignment", "mismatch"],
        "solutions": [
            "Run fitprep to check map/model alignment",
            "Verify map origin and apix values",
            "Check coordinate system (MRC uses Z,Y,X, PDB uses X,Y,Z)",
            "Ensure map and model are in same coordinate frame"
        ],
        "prevention": "Always validate alignment before processing"
    },
    
    "low_quality_results": {
        "patterns": ["low correlation", "poor fit", "bad results", "not working"],
        "solutions": [
            "Check input quality (map resolution, model completeness)",
            "Verify parameters are appropriate for your data",
            "Try different thresholds or search parameters",
            "Validate inputs with fitprep or validate tools",
            "Check if map/model are compatible (right structure?)"
        ],
        "prevention": "Always validate inputs and check intermediate results"
    }
}

# Parameter recommendations based on resolution
RESOLUTION_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "high_resolution": {
        "range": "≤ 2.5 Å",
        "findligands": {
            "threshold": "0.4-0.6",
            "expect_waters": "Many (10-100x more than ions)",
            "ml_filtering": "Aggressive water classification"
        },
        "pathwalker": {
            "threshold": "0.01-0.02",
            "feasible": "Yes, high quality traces possible"
        },
        "foldhunter": {
            "resolution": "Match map resolution",
            "expect_correlation": "> 0.6"
        }
    },
    "medium_resolution": {
        "range": "2.5-3.5 Å",
        "findligands": {
            "threshold": "0.3-0.5",
            "expect_waters": "Few (mostly ions)",
            "ml_filtering": "Conservative, mostly ions"
        },
        "pathwalker": {
            "threshold": "0.02-0.05",
            "feasible": "Yes, but may need pathwalker2"
        },
        "foldhunter": {
            "resolution": "Match map resolution",
            "expect_correlation": "0.4-0.6"
        }
    },
    "low_resolution": {
        "range": "> 3.5 Å",
        "findligands": {
            "threshold": "0.2-0.4",
            "expect_waters": "Very few (almost all ions)",
            "ml_filtering": "Very conservative, filter out waters"
        },
        "pathwalker": {
            "threshold": "0.05-0.1",
            "feasible": "Limited, use pathwalker2 or domain-based"
        },
        "foldhunter": {
            "resolution": "Match map resolution",
            "expect_correlation": "0.3-0.5"
        }
    }
}

