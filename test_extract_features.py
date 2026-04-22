#!/usr/bin/env python3
"""Smoke test: feature extraction on a subset of PDB files from the training bundle.

The large ``TRAINING/`` tree is not kept in this repo. Set:

  export CRYOMODEL_TRAINING_DIR=/path/to/TRAINING

If unset, we look for ``TRAINING/training_set.csv`` next to this file, then
``../CRYOMODEL_LOCAL/TRAINING`` (sibling checkout next to the clone).

Run explicitly (not part of default ``pytest``):

  python test_extract_features.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cryomodel.ml.extract_features import extract_features_batch
import pandas as pd


def _training_root() -> Path:
    env = os.environ.get("CRYOMODEL_TRAINING_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here / "TRAINING",
        here.parent / "CRYOMODEL_LOCAL" / "TRAINING",
    ]
    for c in candidates:
        if (c / "training_set.csv").is_file():
            return c
    raise FileNotFoundError(
        "Could not find training data. Expected training_set.csv under one of:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n"
        "Or set CRYOMODEL_TRAINING_DIR to your TRAINING directory."
    )


def main() -> None:
    root = _training_root()
    training_csv = root / "training_set.csv"
    pdb_dir = root / "PDBs"
    out_csv = root / "test_features.csv"

    training_df = pd.read_csv(training_csv)
    print("Loaded {} structures from training set".format(len(training_df)))

    test_ids = training_df.head(5)["pdb_id"].tolist()
    print("Testing with PDB IDs: {}".format(test_ids))

    result_df = extract_features_batch(
        pdb_dir=pdb_dir,
        output_csv=out_csv,
        pdb_ids=test_ids,
        resolution_csv=training_csv,
        remove_hydrogens=True,
        max_structures=5,
    )

    print("\nExtraction complete!")
    if len(result_df) > 0:
        print("\nClass distribution:")
        print(result_df["label"].value_counts())
        print("\nFeature columns: {}".format(len(result_df.columns)))
        print("Sample features:")
        print(result_df.head())


if __name__ == "__main__":
    main()
