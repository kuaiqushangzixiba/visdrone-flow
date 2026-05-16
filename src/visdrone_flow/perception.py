from __future__ import annotations

import pandas as pd


PERCEPTION_DEFAULTS = {
    "detected_uav_count": 0.0,
    "sensor_confidence": 1.0,
    "simulated_density": 0.0,
    "simulated_avg_speed": 0.0,
    "simulated_em_interference": 0.0,
}


def build_standard_flow_dataset(records: pd.DataFrame, perception: pd.DataFrame | None = None) -> pd.DataFrame:
    """Align historical flight records and simulated perception data.

    The alignment key is the project-wide standard:
    `grid_id + height_layer + time_slot`.
    """

    frame = records.copy()
    frame["grid_id"] = frame["grid_id"].astype(str)
    frame["height_layer"] = frame["height_layer"].astype(int)
    frame["time_slot"] = pd.to_datetime(frame["time_slot"], utc=False)
    if perception is None or perception.empty:
        for column, value in PERCEPTION_DEFAULTS.items():
            if column not in frame.columns:
                frame[column] = value
        return _apply_perception_features(frame)

    sensed = perception.copy()
    sensed["grid_id"] = sensed["grid_id"].astype(str)
    sensed["height_layer"] = sensed["height_layer"].astype(int)
    sensed["time_slot"] = pd.to_datetime(sensed["time_slot"], utc=False)
    for column, value in PERCEPTION_DEFAULTS.items():
        if column not in sensed.columns:
            sensed[column] = value
    sensed = (
        sensed.groupby(["grid_id", "height_layer", "time_slot"], as_index=False)
        .agg(
            {
                "detected_uav_count": "mean",
                "sensor_confidence": "mean",
                "simulated_density": "mean",
                "simulated_avg_speed": "mean",
                "simulated_em_interference": "mean",
            }
        )
    )
    merged = frame.merge(sensed, on=["grid_id", "height_layer", "time_slot"], how="left")
    for column, value in PERCEPTION_DEFAULTS.items():
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(value)
    return _apply_perception_features(merged)


def _apply_perception_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["task_count"] = (
        pd.to_numeric(result.get("task_count", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(result["detected_uav_count"], errors="coerce").fillna(0.0)
        * pd.to_numeric(result["sensor_confidence"], errors="coerce").fillna(1.0)
    )
    result["occupancy"] = pd.concat(
        [
            pd.to_numeric(result.get("occupancy", 0.0), errors="coerce").fillna(0.0),
            pd.to_numeric(result["simulated_density"], errors="coerce").fillna(0.0),
        ],
        axis=1,
    ).max(axis=1)
    simulated_speed = pd.to_numeric(result["simulated_avg_speed"], errors="coerce").fillna(0.0)
    current_speed = pd.to_numeric(result.get("avg_speed", 0.0), errors="coerce").fillna(0.0)
    result["avg_speed"] = current_speed.where(current_speed > 0, simulated_speed)
    result["em_interference"] = pd.concat(
        [
            pd.to_numeric(result.get("em_interference", 0.0), errors="coerce").fillna(0.0),
            pd.to_numeric(result["simulated_em_interference"], errors="coerce").fillna(0.0),
        ],
        axis=1,
    ).max(axis=1)
    return result

