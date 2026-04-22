"""Core services for PathMeasure backend."""
from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path
from typing import List, Optional, Tuple

import mrcfile
import numpy as np

from .models import PathModel, SessionModel


class PathMeasureState:
    """In-memory state for the currently open image."""

    def __init__(self):
        self.image_path: Optional[str] = None
        self.image_2d: Optional[np.ndarray] = None
        self.apix: Optional[float] = None  # authoritative user-entered value
        self.header_apix: Optional[float] = None


def _header_apix_from_mrc(mrc: mrcfile.mrcfile.MrcFile) -> Optional[float]:
    """Best-effort extraction of apix from MRC header."""
    try:
        v = float(mrc.voxel_size.x)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return None


def load_2d_mrc(image_path: str) -> Tuple[np.ndarray, Optional[float]]:
    """Load a 2D MRC image and return data + header apix (if available)."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
        header_apix = _header_apix_from_mrc(mrc)

    if data.ndim == 2:
        img = data
    elif data.ndim == 3 and data.shape[0] == 1:
        img = data[0]
    else:
        raise ValueError(
            f"PathMeasure currently supports 2D MRC only; got shape={tuple(data.shape)}"
        )

    img = np.asarray(img, dtype=np.float32)
    return img, header_apix


def compute_path_length_px(points: List[Tuple[float, float]]) -> float:
    """Compute polyline arc length in pixels."""
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    diffs = arr[1:] - arr[:-1]
    seg = np.sqrt((diffs * diffs).sum(axis=1))
    return float(seg.sum())


def enrich_path_lengths(paths: List[PathModel], apix: float) -> List[PathModel]:
    """Return paths with updated length fields based on current apix."""
    out: List[PathModel] = []
    for p in paths:
        lp = compute_path_length_px(p.points)
        p.length_px = lp
        p.length_angstrom = lp * apix
        out.append(p)
    return out


def save_session_json(session: SessionModel, session_path: str) -> None:
    path = Path(session_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(session.dict(), fh, indent=2)


def load_session_json(session_path: str) -> SessionModel:
    path = Path(session_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SessionModel(**payload)


def export_measurements_csv(paths: List[PathModel], apix: float) -> str:
    """Build CSV text for measurement export."""
    with io.StringIO(newline="") as s:
        writer = csv.writer(s)
        writer.writerow(
            [
                "id",
                "name",
                "group",
                "length_px",
                "length_angstrom",
                "num_points",
                "visible",
                "color",
                "line_width",
                "points_json",
            ]
        )
        for p in enrich_path_lengths(paths, apix):
            writer.writerow(
                [
                    p.id,
                    p.name,
                    p.group,
                    f"{p.length_px:.6f}",
                    f"{p.length_angstrom:.6f}",
                    len(p.points),
                    str(p.visible).lower(),
                    p.color,
                    p.line_width,
                    json.dumps(p.points),
                ]
            )
        return s.getvalue()


def build_preview_u8_b64(
    image_2d: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, max_dim: int = 1600
) -> Tuple[int, int, int, int, float, float, str]:
    """Create a display preview as base64 grayscale uint8 bytes."""
    arr = np.asarray(image_2d, dtype=np.float32)
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    h, w = src_h, src_w

    # Optional downsample for responsiveness.
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        y_idx = np.linspace(0, h - 1, new_h).astype(np.int64)
        x_idx = np.linspace(0, w - 1, new_w).astype(np.int64)
        arr = arr[np.ix_(y_idx, x_idx)]
        h, w = new_h, new_w

    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            hi = lo + 1.0

    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    b64 = base64.b64encode(u8.tobytes()).decode("ascii")
    scale_x = float(src_w) / float(w)
    scale_y = float(src_h) / float(h)
    return w, h, src_w, src_h, scale_x, scale_y, b64

