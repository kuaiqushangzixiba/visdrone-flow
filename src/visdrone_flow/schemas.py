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


class GridCell(BaseModel):
    """Standard low-altitude grid cell used by all non-frontend algorithm modules."""

    model_config = ConfigDict(extra="forbid")

    grid_id: str
    height_layer: int = Field(..., ge=0)
    center_x_m: float
    center_y_m: float
    center_z_m: float = Field(..., ge=0)
    time_slot: datetime | None = None
    route_capacity: float = Field(1, ge=0)
    occupancy: float = Field(0, ge=0)
    congestion_score: float = Field(0, ge=0)
    em_interference: float = Field(0, ge=0)
    weather_wind: float = Field(0, ge=0)
    weather_visibility: float = Field(10_000, ge=0)
    population_density: float = Field(0, ge=0)
    no_fly_flag: int = Field(0, ge=0, le=1)
    risk_score: float = Field(0, ge=0)
    height_ref: HeightReference = "AGL"


class Transmitter(BaseModel):
    """Communication transmitter or jammer candidate."""

    model_config = ConfigDict(extra="forbid")

    transmitter_id: str
    x_m: float
    y_m: float
    z_m: float = Field(..., ge=0)
    frequency_mhz: float = Field(..., gt=0)
    bandwidth_mhz: float = Field(10, gt=0)
    tx_power_dbm: float = 30
    tx_gain_dbi: float = 0
    rx_gain_dbi: float = 0
    noise_figure_db: float = Field(7, ge=0)
    role: Literal["base_station", "relay", "uav", "jammer"] = "base_station"


class UavState(BaseModel):
    """Current UAV state for allocation, routing, and safety modules."""

    model_config = ConfigDict(extra="forbid")

    uav_id: str
    grid_id: str
    height_layer: int = Field(..., ge=0)
    x_m: float
    y_m: float
    z_m: float = Field(..., ge=0)
    vx_mps: float = 0
    vy_mps: float = 0
    vz_mps: float = 0
    speed_mps: float = Field(12, gt=0)
    battery_pct: float = Field(100, ge=0, le=100)
    max_range_m: float = Field(20_000, gt=0)
    payload_capacity_kg: float = Field(0, ge=0)
    current_payload_kg: float = Field(0, ge=0)
    priority: int = Field(1, ge=0)


class MissionTask(BaseModel):
    """Task demand that must be assigned to a UAV and airspace route."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    origin_grid_id: str
    origin_height_layer: int = Field(..., ge=0)
    dest_grid_id: str
    dest_height_layer: int = Field(..., ge=0)
    priority: int = Field(1, ge=0)
    required_payload_kg: float = Field(0, ge=0)
    earliest_time: datetime | None = None
    latest_time: datetime | None = None


class RoutePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_grid_id: str
    start_height_layer: int = Field(..., ge=0)
    end_grid_id: str
    end_height_layer: int = Field(..., ge=0)
    max_nodes: int = Field(10_000, ge=1)

