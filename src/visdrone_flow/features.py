from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grid import make_node_id


DEFAULT_EXOGENOUS_COLUMNS = [
    "flow_out",
    "occupancy",
    "avg_speed",
    "task_count",
    "weather_wind",
    "weather_visibility",
    "em_interference",
    "no_fly_flag",
    "route_capacity",
]


@dataclass(slots=True)
class FeatureConfig:
    target_column: str = "flow_in"
    history_steps: int = 12
    horizon_steps: int = 6
    exogenous_columns: tuple[str, ...] = tuple(DEFAULT_EXOGENOUS_COLUMNS)


@dataclass(slots=True)
class SupervisedData:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    anchor_times: list[pd.Timestamp]
    node_ids: list[str]


def normalize_records(records: pd.DataFrame, exogenous_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = records.copy()
    frame["time_slot"] = pd.to_datetime(frame["time_slot"], utc=False)
    frame["height_layer"] = frame["height_layer"].astype(int)
    frame["node_id"] = [
        make_node_id(grid_id, height_layer)
        for grid_id, height_layer in zip(frame["grid_id"].astype(str), frame["height_layer"])
    ]
    for column in ["flow_in", *exogenous_columns]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.sort_values(["time_slot", "node_id"]).reset_index(drop=True)


def build_neighbor_map(edges: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    neighbors: dict[str, list[tuple[str, float]]] = {}
    if edges is None or edges.empty:
        return neighbors
    for row in edges.itertuples(index=False):
        source = make_node_id(row.source_grid_id, int(row.source_height_layer))
        target = make_node_id(row.target_grid_id, int(row.target_height_layer))
        weight = float(getattr(row, "weight", 1.0))
        neighbors.setdefault(source, []).append((target, weight))
        if not bool(getattr(row, "directed", False)):
            neighbors.setdefault(target, []).append((source, weight))
    return neighbors


class PanelBuilder:
    """Build dense node-time matrices from sparse grid records."""

    def __init__(self, records: pd.DataFrame, config: FeatureConfig):
        self.records = normalize_records(records, config.exogenous_columns)
        self.config = config
        self.times = sorted(self.records["time_slot"].drop_duplicates())
        self.node_ids = sorted(self.records["node_id"].drop_duplicates())
        self.time_index = {time: idx for idx, time in enumerate(self.times)}
        self.node_index = {node: idx for idx, node in enumerate(self.node_ids)}
        self.target_matrix = self._make_matrix(config.target_column)
        self.exog_matrices = {column: self._make_matrix(column) for column in config.exogenous_columns}

    def _make_matrix(self, column: str) -> np.ndarray:
        matrix = np.full((len(self.times), len(self.node_ids)), np.nan, dtype=float)
        for row in self.records[["time_slot", "node_id", column]].itertuples(index=False):
            matrix[self.time_index[row.time_slot], self.node_index[row.node_id]] = float(getattr(row, column))
        if matrix.size == 0:
            return matrix
        column_means = np.nanmean(matrix, axis=0)
        global_mean = float(np.nanmean(matrix)) if not np.isnan(np.nanmean(matrix)) else 0.0
        column_means = np.where(np.isnan(column_means), global_mean, column_means)
        missing = np.where(np.isnan(matrix))
        matrix[missing] = np.take(column_means, missing[1])
        return matrix


def build_supervised(records: pd.DataFrame, edges: pd.DataFrame, config: FeatureConfig) -> SupervisedData:
    panel = PanelBuilder(records, config)
    neighbors = build_neighbor_map(edges)
    h = config.history_steps
    k = config.horizon_steps
    if len(panel.times) < h + k:
        raise ValueError(f"Need at least {h + k} time slots, got {len(panel.times)}")

    feature_names = _feature_names(config)
    samples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchor_times: list[pd.Timestamp] = []
    sample_nodes: list[str] = []

    for t in range(h - 1, len(panel.times) - k):
        for node_id in panel.node_ids:
            node_idx = panel.node_index[node_id]
            samples.append(_features_for(panel, neighbors, node_id, node_idx, t, config))
            targets.append(panel.target_matrix[t + 1 : t + k + 1, node_idx])
            anchor_times.append(panel.times[t])
            sample_nodes.append(node_id)

    return SupervisedData(
        x=np.vstack(samples),
        y=np.vstack(targets),
        feature_names=feature_names,
        anchor_times=anchor_times,
        node_ids=sample_nodes,
    )


def build_prediction_features(
    records: pd.DataFrame,
    edges: pd.DataFrame,
    config: FeatureConfig,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.Timestamp]:
    panel = PanelBuilder(records, config)
    if len(panel.times) < config.history_steps:
        raise ValueError(f"Need at least {config.history_steps} recent time slots, got {len(panel.times)}")
    neighbors = build_neighbor_map(edges)
    anchor_t = len(panel.times) - 1
    features = [
        _features_for(panel, neighbors, node_id, panel.node_index[node_id], anchor_t, config)
        for node_id in panel.node_ids
    ]
    if feature_names is not None and feature_names != _feature_names(config):
        raise ValueError("Feature config does not match trained model")
    return np.vstack(features), panel.node_ids, panel.times[anchor_t]


def _feature_names(config: FeatureConfig) -> list[str]:
    names: list[str] = []
    for lag in range(1, config.history_steps + 1):
        names.append(f"{config.target_column}_lag_{lag}")
    for lag in range(1, min(config.history_steps, 3) + 1):
        names.append(f"neighbor_{config.target_column}_lag_{lag}")
    for column in config.exogenous_columns:
        names.append(f"{column}_lag_1")
    names.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    return names


def _features_for(
    panel: PanelBuilder,
    neighbors: dict[str, list[tuple[str, float]]],
    node_id: str,
    node_idx: int,
    t: int,
    config: FeatureConfig,
) -> np.ndarray:
    values: list[float] = []
    for lag in range(1, config.history_steps + 1):
        values.append(float(panel.target_matrix[t - lag + 1, node_idx]))
    for lag in range(1, min(config.history_steps, 3) + 1):
        values.append(_neighbor_mean(panel, neighbors, node_id, t - lag + 1))
    for column in config.exogenous_columns:
        values.append(float(panel.exog_matrices[column][t, node_idx]))
    timestamp = pd.Timestamp(panel.times[t])
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    values.append(float(np.sin(2 * np.pi * minute_of_day / 1440)))
    values.append(float(np.cos(2 * np.pi * minute_of_day / 1440)))
    values.append(float(np.sin(2 * np.pi * timestamp.dayofweek / 7)))
    values.append(float(np.cos(2 * np.pi * timestamp.dayofweek / 7)))
    return np.asarray(values, dtype=float)


def _neighbor_mean(
    panel: PanelBuilder,
    neighbors: dict[str, list[tuple[str, float]]],
    node_id: str,
    time_idx: int,
) -> float:
    linked = neighbors.get(node_id)
    if not linked:
        return float(panel.target_matrix[time_idx].mean())
    weighted_sum = 0.0
    weight_total = 0.0
    for target_node, weight in linked:
        target_idx = panel.node_index.get(target_node)
        if target_idx is None:
            continue
        weighted_sum += float(panel.target_matrix[time_idx, target_idx]) * weight
        weight_total += weight
    if weight_total <= 0:
        return float(panel.target_matrix[time_idx].mean())
    return weighted_sum / weight_total

