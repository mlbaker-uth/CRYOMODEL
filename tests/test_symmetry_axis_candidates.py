"""Unit tests for symmetry phase-1 axis candidates."""
from __future__ import annotations

import numpy as np

from cryomodel.symmetry.axis_candidates import (
    canonical_axis_key,
    merge_candidate_sources,
    run_phase1_candidates,
)


def test_canonical_axis_key_identifies_opposites():
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, -1.0])
    assert canonical_axis_key(a) == canonical_axis_key(b)


def test_merge_has_cardinal_pca_and_reasonable_count():
    # Tilted axes so they are not deduped away as duplicates of cardinals/diagonals.
    pca = [[0.97, 0.242, 0.0], [-0.242, 0.97, 0.0], [0.0, 0.0, 1.0]]
    m = merge_candidate_sources((0.0, 10.0), include_diagonals=True, pca_axes=pca)
    sources = {s for _, s in m}
    assert "cardinal_tilt" in sources
    assert "diagonal" in sources
    assert any(s.startswith("pca_axis") for s in sources)
    assert len(m) >= 20


def test_phase1_requires_phase0_artifacts(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        run_phase1_candidates(tmp_path)
