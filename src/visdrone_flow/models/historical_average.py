from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import FeatureConfig, normalize_records
from ..grid import make_node_id


@dataclass(slots=True)
class HistoricalAverageModel:
    """Fallback model based on node and hour-of-day historical averages."""

    config: FeatureConfig
    by_node_hour: dict[tuple[str, int], float]
    by_node: dict[str, float]
    global_mean: float

    @classmethod
    def fit(cls, records: pd.DataFrame, config: FeatureConfig) -> "HistoricalAverageModel":
        frame = normalize_records(records, config.exogenous_columns)
        frame["hour"] = frame["time_slot"].dt.hour
        by_node_hour = (
            frame.groupby(["node_id", "hour"])[config.target_column].mean().to_dict()
        )
        by_node = frame.groupby("node_id")[config.target_column].mean().to_dict()
        global_mean = float(frame[config.target_column].mean())
        return cls(config=config, by_node_hour=by_node_hour, by_node=by_node, global_mean=global_mean)

    def predict(self, records: pd.DataFrame, horizon_steps: int) -> tuple[np.ndarray, list[str], pd.Timestamp]:
        frame = normalize_records(records, self.config.exogenous_columns)
        latest_time = frame["time_slot"].max()
        node_ids = sorted(frame["node_id"].drop_duplicates())
        predictions = np.zeros((len(node_ids), horizon_steps), dtype=float)
        step = _infer_step(frame["time_slot"].drop_duplicates().sort_values())
        for node_idx, node_id in enumerate(node_ids):
            for horizon in range(horizon_steps):
                future_time = latest_time + step * (horizon + 1)
                value = self.by_node_hour.get((node_id, future_time.hour))
                if value is None:
                    value = self.by_node.get(node_id, self.global_mean)
                predictions[node_idx, horizon] = max(0.0, float(value))
        return predictions, node_ids, pd.Timestamp(latest_time)

    def predict_node_value(self, grid_id: str, height_layer: int, timestamp: pd.Timestamp) -> float:
        node_id = make_node_id(grid_id, height_layer)
        return float(self.by_node_hour.get((node_id, timestamp.hour), self.by_node.get(node_id, self.global_mean)))


def _infer_step(times: pd.Series) -> pd.Timedelta:
    values = list(pd.to_datetime(times))
    if len(values) < 2:
        return pd.Timedelta(minutes=5)
    diffs = pd.Series(values).diff().dropna()
    return pd.Timedelta(diffs.mode().iloc[0]) if not diffs.empty else pd.Timedelta(minutes=5)

