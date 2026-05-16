from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import FlowRecord, GridEdge


REQUIRED_RECORD_COLUMNS = {
    "grid_id",
    "height_layer",
    "time_slot",
    "flow_in",
}


def read_records_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_RECORD_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required record columns: {sorted(missing)}")
    frame["time_slot"] = pd.to_datetime(frame["time_slot"], utc=False)
    frame["height_layer"] = frame["height_layer"].astype(int)
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame.sort_values(["time_slot", "grid_id", "height_layer"]).reset_index(drop=True)


def read_edges_csv(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(
            columns=[
                "source_grid_id",
                "source_height_layer",
                "target_grid_id",
                "target_height_layer",
                "weight",
                "edge_type",
                "directed",
            ]
        )
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for column in ["source_height_layer", "target_height_layer"]:
        frame[column] = frame[column].astype(int)
    if "weight" not in frame.columns:
        frame["weight"] = 1.0
    if "edge_type" not in frame.columns:
        frame["edge_type"] = "adjacent"
    if "directed" not in frame.columns:
        frame["directed"] = False
    return frame


def read_optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_csv(path)


def records_from_models(records: list[FlowRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump(mode="json") for record in records]).assign(
        time_slot=lambda df: pd.to_datetime(df["time_slot"], utc=False)
    )


def edges_from_models(edges: list[GridEdge]) -> pd.DataFrame:
    if not edges:
        return read_edges_csv(None)
    return pd.DataFrame([edge.model_dump(mode="json") for edge in edges])


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
