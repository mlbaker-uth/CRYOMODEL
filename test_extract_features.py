#!/usr/bin/env python3
"""Test feature extraction on a subset of PDB files."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from crymodel.ml.extract_features import extract_features_batch
import pandas as pd

# Load training set to get PDB IDs and resolutions
training_df = pd.read_csv("TRAINING/training_set.csv")
print("Loaded {} structures from training set".format(len(training_df)))

# Get first 5 PDB IDs for testing
test_ids = training_df.head(5)["pdb_id"].tolist()
print("Testing with PDB IDs: {}".format(test_ids))

# Extract features
result_df = extract_features_batch(
    pdb_dir=Path("TRAINING/PDBs"),
    output_csv=Path("TRAINING/test_features.csv"),
    pdb_ids=test_ids,
    resolution_csv=Path("TRAINING/training_set.csv"),
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

