from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

@dataclass
class MapVolume:
    data: np.ndarray
    apix: float
    origin: Tuple[float,float,float] = (0.0, 0.0, 0.0)
    halfmaps: Optional[Tuple[np.ndarray, np.ndarray]] = None

@dataclass
class ModelAtoms:
    xyz: np.ndarray; name: np.ndarray; resname: np.ndarray
    chain: np.ndarray; resi: np.ndarray; element: np.ndarray

@dataclass
class PseudoAtoms:
    xyz: np.ndarray
    cluster_id: Optional[np.ndarray] = None
    score: Optional[np.ndarray] = None

@dataclass
class FragmentPose:
    center: np.ndarray
    R: np.ndarray
    torsions: Optional[np.ndarray] = None

@dataclass
class Assignment:
    index: int; cluster_id: int
    probs: Dict[str, float]; top: str
    explain: Dict[str, Any]

@dataclass
class AssignmentSet:
    centers: np.ndarray
    assignments: List[Assignment]
    meta: Dict[str, Any]
