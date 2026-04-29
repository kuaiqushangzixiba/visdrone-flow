from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .grid import split_node_id
from .io import edges_from_models, records_from_models
from .model_store import ModelBundle
from .schemas import PredictionPoint, PredictionRequest, PredictionResponse, WarningLevel


class FlowForecastService:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        records = records_from_models(request.records)
        edges = edges_from_models(request.edges)
        model = self.bundle.model
        values, node_ids, anchor_time = model.predict(records, edges, request.horizon_steps)
        latest_records = _latest_by_node(records)
        predictions = _to_points(
            values=values,
            node_ids=node_ids,
            anchor_time=anchor_time,
            latest_records=latest_records,
            horizon_steps=request.horizon_steps,
            minutes=request.time_granularity_minutes,
            target_mean=float(getattr(model, "target_mean", np.mean(values) if values.size else 0.0)),
            target_std=float(getattr(model, "target_std", np.std(values) or 1.0)),
        )
        return PredictionResponse(
            request_id=request.request_id,
            model_name=self.bundle.model_name,
            generated_at=datetime.now(timezone.utc),
            horizon_steps=request.horizon_steps,
            time_granularity_minutes=request.time_granularity_minutes,
            predictions=predictions,
        )


def _latest_by_node(records: pd.DataFrame) -> dict[str, pd.Series]:
    frame = records.copy()
    frame["time_slot"] = pd.to_datetime(frame["time_slot"], utc=False)
    frame["node_id"] = frame["grid_id"].astype(str) + "#H" + frame["height_layer"].astype(str)
    latest = frame.sort_values("time_slot").groupby("node_id", as_index=False).tail(1)
    return {row.node_id: row for row in latest.itertuples(index=False)}


def _to_points(
    values: np.ndarray,
    node_ids: list[str],
    anchor_time: pd.Timestamp,
    latest_records: dict[str, pd.Series],
    horizon_steps: int,
    minutes: int,
    target_mean: float,
    target_std: float,
) -> list[PredictionPoint]:
    points: list[PredictionPoint] = []
    for node_idx, node_id in enumerate(node_ids):
        node = split_node_id(node_id)
        latest = latest_records.get(node_id)
        capacity = float(getattr(latest, "route_capacity", 1.0) or 1.0)
        capacity = max(capacity, 1e-6)
        for horizon in range(horizon_steps):
            pred_flow = float(values[node_idx, horizon])
            future_time = pd.Timestamp(anchor_time).to_pydatetime() + timedelta(minutes=minutes * (horizon + 1))
            density = pred_flow / capacity
            congestion_score = max(0.0, density)
            points.append(
                PredictionPoint(
                    grid_id=node.grid_id,
                    height_layer=node.height_layer,
                    future_time_slot=future_time,
                    horizon_index=horizon + 1,
                    pred_flow=round(pred_flow, 6),
                    pred_density=round(density, 6),
                    congestion_score=round(congestion_score, 6),
                    warning_level=_warning_level(congestion_score),
                    confidence=_confidence(pred_flow, target_mean, target_std, horizon + 1),
                )
            )
    return points


def _warning_level(congestion_score: float) -> WarningLevel:
    if congestion_score >= 1.2:
        return WarningLevel.severe
    if congestion_score >= 0.85:
        return WarningLevel.congested
    if congestion_score >= 0.65:
        return WarningLevel.attention
    return WarningLevel.normal


def _confidence(value: float, mean: float, std: float, horizon_index: int) -> float:
    z = abs(value - mean) / max(std, 1e-6)
    horizon_penalty = min(0.35, 0.035 * (horizon_index - 1))
    confidence = 0.92 - min(0.35, z * 0.08) - horizon_penalty
    return round(float(min(0.98, max(0.2, confidence))), 6)

