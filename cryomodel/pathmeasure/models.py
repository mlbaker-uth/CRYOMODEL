"""Data models for PathMeasure backend."""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


Point2D = Tuple[float, float]


class PathModel(BaseModel):
    id: str
    name: str
    group: str = "default"
    points: List[Point2D] = Field(default_factory=list)
    color: str = "#ff0000"
    line_width: float = 2.0
    visible: bool = True
    locked: bool = False
    creation_mode: str = "polyline"  # polyline | freehand
    length_px: Optional[float] = None
    length_angstrom: Optional[float] = None


class DisplaySettings(BaseModel):
    min_intensity: Optional[float] = None
    max_intensity: Optional[float] = None
    gamma: float = 1.0
    invert: bool = False
    auto_percentile_low: float = 1.0
    auto_percentile_high: float = 99.0


class SessionModel(BaseModel):
    session_version: str = "0.1.0"
    image_path: str
    image_width: int
    image_height: int
    apix: float  # authoritative user-entered calibration
    header_apix: Optional[float] = None
    paths: List[PathModel] = Field(default_factory=list)
    display_settings: DisplaySettings = Field(default_factory=DisplaySettings)


class OpenImageRequest(BaseModel):
    image_path: str
    apix: float = Field(..., gt=0.0, description="User-entered Angstrom per pixel")


class OpenImageResponse(BaseModel):
    image_path: str
    width: int
    height: int
    dtype: str
    apix: float
    header_apix: Optional[float] = None
    min_intensity: float
    max_intensity: float
    mean_intensity: float
    std_intensity: float


class ImageMetadataResponse(BaseModel):
    image_path: str
    width: int
    height: int
    apix: float
    header_apix: Optional[float] = None


class ImagePreviewResponse(BaseModel):
    width: int
    height: int
    source_width: int
    source_height: int
    scale_x: float  # source_width / width
    scale_y: float  # source_height / height
    pixels_u8_b64: str  # grayscale row-major uint8 bytes, base64 encoded


class SaveSessionRequest(BaseModel):
    session_path: str
    session: SessionModel


class LoadSessionRequest(BaseModel):
    session_path: str


class ExportCSVRequest(BaseModel):
    paths: List[PathModel]
    apix: float = Field(..., gt=0.0)
    output_csv: Optional[str] = None


class ExportCSVResponse(BaseModel):
    rows_written: int
    output_csv: Optional[str] = None
    csv_text: str

