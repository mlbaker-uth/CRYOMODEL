"""State container for interactive BaseHunter pair/marker workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PAIR_STATES = {"new", "ready", "computed", "stale", "error"}


@dataclass
class Marker3D:
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @staticmethod
    def from_dict(payload: Dict[str, float]) -> "Marker3D":
        return Marker3D(float(payload["x"]), float(payload["y"]), float(payload["z"]))


@dataclass
class PairResult:
    pA_purine: float
    pA_pyrimidine: float
    pB_purine: float
    pB_pyrimidine: float
    joint_wc: float
    planarity_rms: float
    clash_metric: float
    confidence: float
    call: str

    def to_dict(self) -> Dict[str, float]:
        return {
            "pA_purine": self.pA_purine,
            "pA_pyrimidine": self.pA_pyrimidine,
            "pB_purine": self.pB_purine,
            "pB_pyrimidine": self.pB_pyrimidine,
            "joint_wc": self.joint_wc,
            "planarity_rms": self.planarity_rms,
            "clash_metric": self.clash_metric,
            "confidence": self.confidence,
            "call": self.call,
        }

    @staticmethod
    def from_dict(payload: Dict[str, float]) -> "PairResult":
        return PairResult(
            pA_purine=float(payload["pA_purine"]),
            pA_pyrimidine=float(payload["pA_pyrimidine"]),
            pB_purine=float(payload["pB_purine"]),
            pB_pyrimidine=float(payload["pB_pyrimidine"]),
            joint_wc=float(payload["joint_wc"]),
            planarity_rms=float(payload["planarity_rms"]),
            clash_metric=float(payload["clash_metric"]),
            confidence=float(payload["confidence"]),
            call=str(payload["call"]),
        )


@dataclass
class PairRecord:
    pair_id: str
    marker_a: Optional[Marker3D] = None
    marker_b: Optional[Marker3D] = None
    status: str = "new"
    label: str = ""
    result: Optional[PairResult] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "marker_a": None if self.marker_a is None else self.marker_a.to_dict(),
            "marker_b": None if self.marker_b is None else self.marker_b.to_dict(),
            "status": self.status,
            "label": self.label,
            "result": None if self.result is None else self.result.to_dict(),
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "PairRecord":
        rec = PairRecord(
            pair_id=str(payload["pair_id"]),
            marker_a=Marker3D.from_dict(payload["marker_a"]) if payload.get("marker_a") else None,
            marker_b=Marker3D.from_dict(payload["marker_b"]) if payload.get("marker_b") else None,
            status=str(payload.get("status", "new")),
            label=str(payload.get("label", "")),
            result=PairResult.from_dict(payload["result"]) if payload.get("result") else None,
        )
        rec.status = rec.status if rec.status in PAIR_STATES else "error"
        return rec


@dataclass
class PairState:
    pairs: List[PairRecord] = field(default_factory=list)
    selected_pair_id: Optional[str] = None
    marker_mode: str = "place"
    auto_pair: bool = True
    _counter: int = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"BP_{self._counter:03d}"

    def add_pair(self, label: str = "") -> PairRecord:
        pair = PairRecord(pair_id=self._next_id(), label=label)
        self.pairs.append(pair)
        self.selected_pair_id = pair.pair_id
        return pair

    def get(self, pair_id: Optional[str]) -> Optional[PairRecord]:
        if not pair_id:
            return None
        for p in self.pairs:
            if p.pair_id == pair_id:
                return p
        return None

    def get_selected(self) -> Optional[PairRecord]:
        return self.get(self.selected_pair_id)

    def clear_all(self) -> None:
        self.pairs.clear()
        self.selected_pair_id = None

    def clear_selected(self) -> bool:
        p = self.get_selected()
        if p is None:
            return False
        self.pairs = [x for x in self.pairs if x.pair_id != p.pair_id]
        self.selected_pair_id = self.pairs[0].pair_id if self.pairs else None
        return True

    def add_marker(self, marker: Marker3D, side: Optional[str] = None) -> Optional[PairRecord]:
        p = self.get_selected()
        if p is None:
            p = self.add_pair()
        target_side = side
        if target_side is None:
            if p.marker_a is None:
                target_side = "A"
            elif p.marker_b is None:
                target_side = "B"
            elif self.auto_pair:
                p = self.add_pair()
                target_side = "A"
            else:
                target_side = "B"
        if target_side.upper() == "A":
            p.marker_a = marker
        else:
            p.marker_b = marker
        if p.status == "computed":
            p.status = "stale"
        elif p.marker_a is not None and p.marker_b is not None:
            p.status = "ready"
        else:
            p.status = "new"
        p.result = None
        self.selected_pair_id = p.pair_id
        return p

    def mark_computed(self, pair_id: str, result: PairResult) -> None:
        p = self.get(pair_id)
        if p is None:
            return
        p.result = result
        p.status = "computed"

    def set_selected(self, pair_id: Optional[str]) -> None:
        if pair_id is None:
            self.selected_pair_id = None
        elif self.get(pair_id) is not None:
            self.selected_pair_id = pair_id

    def to_dict(self) -> Dict[str, object]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "selected_pair_id": self.selected_pair_id,
            "marker_mode": self.marker_mode,
            "auto_pair": self.auto_pair,
            "counter": self._counter,
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "PairState":
        st = PairState(
            pairs=[PairRecord.from_dict(x) for x in payload.get("pairs", [])],
            selected_pair_id=payload.get("selected_pair_id"),
            marker_mode=str(payload.get("marker_mode", "place")),
            auto_pair=bool(payload.get("auto_pair", True)),
            _counter=int(payload.get("counter", 0)),
        )
        if st.selected_pair_id and st.get(st.selected_pair_id) is None:
            st.selected_pair_id = st.pairs[0].pair_id if st.pairs else None
        if st._counter < len(st.pairs):
            st._counter = len(st.pairs)
        return st

