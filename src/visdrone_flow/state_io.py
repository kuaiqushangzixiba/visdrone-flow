from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_cells_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"grid_id", "height_layer", "center_x_m", "center_y_m", "center_z_m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required cell columns: {sorted(missing)}")
    frame["grid_id"] = frame["grid_id"].astype(str)
    frame["height_layer"] = frame["height_layer"].astype(int)
    defaults = {
        "route_capacity": 1.0,
        "occupancy": 0.0,
        "congestion_score": 0.0,
        "em_interference": 0.0,
        "weather_wind": 0.0,
        "weather_visibility": 10_000.0,
        "population_density": 0.0,
        "no_fly_flag": 0,
        "risk_score": 0.0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(value)
    return frame


def read_transmitters_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"transmitter_id", "x_m", "y_m", "z_m", "frequency_mhz"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required transmitter columns: {sorted(missing)}")
    defaults = {
        "bandwidth_mhz": 10,
        "tx_power_dbm": 30,
        "tx_gain_dbi": 0,
        "rx_gain_dbi": 0,
        "noise_figure_db": 7,
        "role": "base_station",
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    return frame


def read_uavs_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"uav_id", "grid_id", "height_layer", "x_m", "y_m", "z_m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required uav columns: {sorted(missing)}")
    defaults = {
        "vx_mps": 0,
        "vy_mps": 0,
        "vz_mps": 0,
        "speed_mps": 12,
        "battery_pct": 100,
        "max_range_m": 20_000,
        "payload_capacity_kg": 0,
        "current_payload_kg": 0,
        "priority": 1,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    frame["height_layer"] = frame["height_layer"].astype(int)
    return frame


def read_tasks_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "task_id",
        "origin_grid_id",
        "origin_height_layer",
        "dest_grid_id",
        "dest_height_layer",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required task columns: {sorted(missing)}")
    defaults = {"priority": 1, "required_payload_kg": 0}
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    for column in ["origin_height_layer", "dest_height_layer"]:
        frame[column] = frame[column].astype(int)
    return frame
