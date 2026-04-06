"""
schemas.py — Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DetectionResponse(BaseModel):
    """
    Response payload for POST /detect.

    The annotated image is returned as a base64-encoded PNG string so the
    client receives everything in a single JSON payload without needing
    multipart parsing on the response side.
    """

    annotated_image_base64: str = Field(
        ...,
        description="Base64-encoded PNG of the blueprint with walls and rooms annotated"
    )
    wall_segment_count: int = Field(
        ...,
        description=(
            "Number of Hough line segments detected and classified as walls. "
            "Note: a single architectural wall may produce multiple segments."
        )
    )
    room_count: int = Field(
        ...,
        description="Number of distinct rooms identified"
    )
    room_areas_px: list[int] = Field(
        default_factory=list,
        description="Area in pixels for each detected room, ordered by label"
    )
    image_width: int = Field(..., description="Width of the input image in pixels")
    image_height: int = Field(..., description="Height of the input image in pixels")
    processing_time_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "annotated_image_base64": "<base64 string>",
                "wall_segment_count": 42,
                "room_count": 5,
                "room_areas_px": [12000, 8500, 6200, 4100, 3300],
                "image_width": 1024,
                "image_height": 768,
                "processing_time_ms": 312.4,
            }
        }


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
