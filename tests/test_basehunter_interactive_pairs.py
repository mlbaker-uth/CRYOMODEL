from __future__ import annotations

from cryomodel.nucleotide.interactive_pairs import Marker3D, PairResult, PairState


def test_pair_state_auto_pair_markers() -> None:
    st = PairState(auto_pair=True)
    p1 = st.add_marker(Marker3D(1, 2, 3))
    assert p1 is not None
    assert p1.marker_a is not None
    assert p1.marker_b is None
    assert p1.status == "new"

    p1b = st.add_marker(Marker3D(4, 5, 6))
    assert p1b is not None and p1b.pair_id == p1.pair_id
    assert p1b.marker_b is not None
    assert p1b.status == "ready"

    p2 = st.add_marker(Marker3D(7, 8, 9))
    assert p2 is not None
    assert p2.pair_id != p1.pair_id
    assert p2.marker_a is not None
    assert p2.marker_b is None


def test_pair_state_roundtrip_and_compute() -> None:
    st = PairState()
    p = st.add_pair()
    st.add_marker(Marker3D(0, 0, 0), side="A")
    st.add_marker(Marker3D(1, 1, 1), side="B")
    assert p.status == "ready"
    st.mark_computed(
        p.pair_id,
        PairResult(
            pA_purine=0.8,
            pA_pyrimidine=0.2,
            pB_purine=0.1,
            pB_pyrimidine=0.9,
            joint_wc=0.95,
            planarity_rms=0.2,
            clash_metric=0.1,
            confidence=0.93,
            call="A-T",
        ),
    )
    assert p.status == "computed"
    payload = st.to_dict()
    st2 = PairState.from_dict(payload)
    p2 = st2.get(p.pair_id)
    assert p2 is not None
    assert p2.status == "computed"
    assert p2.result is not None
    assert abs(p2.result.joint_wc - 0.95) < 1e-9

