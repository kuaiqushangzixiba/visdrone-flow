from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


HeightReference = Literal["AGL", "MSL", "ELLIPSOID"]


class WarningLevel(IntEnum):
    normal = 0
    attention = 1
    congested = 2
    severe = 3


class FlowRecord(BaseModel):
    """One observed low-altitude traffic record for a grid cell and time slot."""

    model_config = ConfigDict(extra="forbid")

    grid_id: str = Field(..., description="External standard grid code, e.g. GB/T 39409 code.")
    height_layer: int = Field(..., ge=0)
    time_slot: datetime
    flow_in: float = Field(..., ge=0)
    flow_out: float = Field(0, ge=0)
    occupancy: float = Field(0, ge=0)
    avg_speed: float = Field(0, ge=0)
    task_count: float = Field(0, ge=0)
    weather_wind: float = Field(0, ge=0)
    weather_visibility: float = Field(10_000, ge=0)
    em_interference: float = Field(0, ge=0)
    no_fly_flag: int = Field(0, ge=0, le=1)
    route_capacity: float = Field(1, ge=0)
    height_ref: HeightReference = "AGL"

    @field_validator("grid_id")
    @classmethod
    def validate_grid_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("grid_id cannot be empty")
        return value


class GridEdge(BaseModel):
    """Directed or undirected edge between low-altitude grid nodes."""

    model_config = ConfigDict(extra="forbid")

    source_grid_id: str
    source_height_layer: int = Field(..., ge=0)
    target_grid_id: str
    target_height_layer: int = Field(..., ge=0)
    weight: float = Field(1.0, gt=0)
    edge_type: Literal["adjacent", "vertical", "route", "correlation"] = "adjacent"
    directed: bool = False


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = "default"
    horizon_steps: int = Field(6, ge=1, le=288)
    time_granularity_minutes: int = Field(5, ge=1, le=1440)
    records: list[FlowRecord]
    edges: list[GridEdge] = Field(default_factory=list)


class PredictionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_id: str
    height_layer: int
    future_time_slot: datetime
    horizon_index: int = Field(..., ge=1)
    pred_flow: float
    pred_density: float
    congestion_score: float = Field(..., ge=0)
    warning_level: WarningLevel
    confidence: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    model_name: str
    generated_at: datetime
    horizon_steps: int
    time_granularity_minutes: int
    predictions: list[PredictionPoint]

