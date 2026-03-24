"""FastAPI app for PathMeasure (first-pass backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ExportCSVRequest,
    ExportCSVResponse,
    ImagePreviewResponse,
    ImageMetadataResponse,
    LoadSessionRequest,
    OpenImageRequest,
    OpenImageResponse,
    SaveSessionRequest,
    SessionModel,
)
from .service import (
    PathMeasureState,
    build_preview_u8_b64,
    enrich_path_lengths,
    export_measurements_csv,
    load_2d_mrc,
    load_session_json,
    save_session_json,
)

app = FastAPI(title="CryoModel PathMeasure API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
STATE = PathMeasureState()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/open-image", response_model=OpenImageResponse)
def open_image(req: OpenImageRequest) -> OpenImageResponse:
    try:
        image, header_apix = load_2d_mrc(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

    STATE.image_path = str(Path(req.image_path).expanduser().resolve())
    STATE.image_2d = image
    STATE.apix = float(req.apix)  # authoritative user-entered calibration
    STATE.header_apix = header_apix

    return OpenImageResponse(
        image_path=STATE.image_path,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        dtype=str(image.dtype),
        apix=STATE.apix,
        header_apix=header_apix,
        min_intensity=float(np.min(image)),
        max_intensity=float(np.max(image)),
        mean_intensity=float(np.mean(image)),
        std_intensity=float(np.std(image)),
    )


@app.get("/image-metadata", response_model=ImageMetadataResponse)
def image_metadata() -> ImageMetadataResponse:
    if STATE.image_2d is None or STATE.image_path is None or STATE.apix is None:
        raise HTTPException(status_code=400, detail="No image loaded. Call /open-image first.")
    return ImageMetadataResponse(
        image_path=STATE.image_path,
        width=int(STATE.image_2d.shape[1]),
        height=int(STATE.image_2d.shape[0]),
        apix=float(STATE.apix),
        header_apix=STATE.header_apix,
    )


@app.get("/image-preview", response_model=ImagePreviewResponse)
def image_preview(
    p_low: float = 1.0,
    p_high: float = 99.0,
    max_dim: int = 1600,
) -> ImagePreviewResponse:
    if STATE.image_2d is None:
        raise HTTPException(status_code=400, detail="No image loaded. Call /open-image first.")
    if not (0.0 <= p_low < p_high <= 100.0):
        raise HTTPException(status_code=400, detail="Percentiles must satisfy 0 <= p_low < p_high <= 100.")
    try:
        w, h, src_w, src_h, sx, sy, b64 = build_preview_u8_b64(
            STATE.image_2d, p_low=p_low, p_high=p_high, max_dim=max_dim
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {e}")
    return ImagePreviewResponse(
        width=w,
        height=h,
        source_width=src_w,
        source_height=src_h,
        scale_x=sx,
        scale_y=sy,
        pixels_u8_b64=b64,
    )


@app.post("/save-session")
def save_session(req: SaveSessionRequest) -> Dict[str, str]:
    try:
        session = req.session
        session.paths = enrich_path_lengths(session.paths, session.apix)
        save_session_json(session, req.session_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {e}")
    return {"status": "saved", "session_path": str(Path(req.session_path).expanduser().resolve())}


@app.post("/load-session", response_model=SessionModel)
def load_session(req: LoadSessionRequest) -> SessionModel:
    try:
        session = load_session_json(req.session_path)
        session.paths = enrich_path_lengths(session.paths, session.apix)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {req.session_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")
    return session


@app.post("/export-csv", response_model=ExportCSVResponse)
def export_csv(req: ExportCSVRequest) -> ExportCSVResponse:
    try:
        csv_text = export_measurements_csv(req.paths, req.apix)
        output_csv = None
        if req.output_csv:
            out_path = Path(req.output_csv).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(csv_text, encoding="utf-8")
            output_csv = str(out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {e}")
    return ExportCSVResponse(
        rows_written=len(req.paths),
        output_csv=output_csv,
        csv_text=csv_text,
    )

