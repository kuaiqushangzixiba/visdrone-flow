from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(
    records_path: str | Path,
    edges_path: str | Path,
    seed: int = 42,
    perception_path: str | Path | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    perception_rows = []
    start = pd.Timestamp("2026-04-29T08:00:00")
    grid_rows = range(4)
    grid_cols = range(4)
    height_layers = range(3)
    periods = 7 * 24 * 12

    for step in range(periods):
        timestamp = start + pd.Timedelta(minutes=5 * step)
        hour_factor = 1.0 + 0.45 * np.sin(2 * np.pi * (timestamp.hour * 60 + timestamp.minute) / 1440)
        rush = 1.0 + (0.35 if timestamp.hour in (9, 10, 17, 18) else 0.0)
        wind = max(0.0, 3.0 + 1.5 * np.sin(step / 40) + rng.normal(0, 0.4))
        visibility = max(500.0, 9000 + 900 * np.cos(step / 60) + rng.normal(0, 200))
        for r in grid_rows:
            for c in grid_cols:
                for h in height_layers:
                    grid_id = f"BDG-L18-R{r:02d}-C{c:02d}"
                    spatial = 1.0 + 0.08 * r + 0.05 * c + 0.12 * h
                    corridor_boost = 1.35 if c in (1, 2) and h == 1 else 1.0
                    weather_penalty = max(0.25, 1.0 - wind * 0.025)
                    base = 7.0 * spatial * corridor_boost * hour_factor * rush * weather_penalty
                    task_count = max(0.0, base * 0.45 + rng.normal(0, 1.2))
                    flow = max(0.0, base + 0.7 * task_count + rng.normal(0, 1.5))
                    capacity = 24 + 4 * h + (6 if c in (1, 2) else 0)
                    rows.append(
                        {
                            "grid_id": grid_id,
                            "height_layer": h,
                            "time_slot": timestamp.isoformat(),
                            "flow_in": round(flow, 4),
                            "flow_out": round(max(0.0, flow + rng.normal(0, 1.1)), 4),
                            "occupancy": round(min(1.5, flow / capacity), 4),
                            "avg_speed": round(max(2.0, 14.0 - 0.15 * flow + rng.normal(0, 0.7)), 4),
                            "task_count": round(task_count, 4),
                            "weather_wind": round(wind, 4),
                            "weather_visibility": round(visibility, 4),
                            "em_interference": round(max(0.0, 0.2 + 0.05 * c + rng.normal(0, 0.03)), 4),
                            "no_fly_flag": 0,
                            "route_capacity": capacity,
                            "height_ref": "AGL",
                        }
                    )
                    if step % 3 == 0:
                        perception_rows.append(
                            {
                                "grid_id": grid_id,
                                "height_layer": h,
                                "time_slot": timestamp.isoformat(),
                                "detected_uav_count": round(max(0.0, task_count + rng.normal(0, 0.6)), 4),
                                "sensor_confidence": round(float(np.clip(0.85 + rng.normal(0, 0.05), 0.5, 1.0)), 4),
                                "simulated_density": round(min(1.5, flow / capacity + rng.normal(0, 0.03)), 4),
                                "simulated_avg_speed": round(max(2.0, 14.0 - 0.14 * flow + rng.normal(0, 0.5)), 4),
                                "simulated_em_interference": round(max(0.0, 0.18 + 0.05 * c + rng.normal(0, 0.04)), 4),
                            }
                        )

    edge_rows = []
    for r in grid_rows:
        for c in grid_cols:
            for h in height_layers:
                source = f"BDG-L18-R{r:02d}-C{c:02d}"
                for dr, dc, edge_type in [(1, 0, "adjacent"), (0, 1, "route")]:
                    rr, cc = r + dr, c + dc
                    if rr in grid_rows and cc in grid_cols:
                        edge_rows.append(
                            {
                                "source_grid_id": source,
                                "source_height_layer": h,
                                "target_grid_id": f"BDG-L18-R{rr:02d}-C{cc:02d}",
                                "target_height_layer": h,
                                "weight": 1.0,
                                "edge_type": edge_type,
                                "directed": False,
                            }
                        )
                if h + 1 in height_layers:
                    edge_rows.append(
                        {
                            "source_grid_id": source,
                            "source_height_layer": h,
                            "target_grid_id": source,
                            "target_height_layer": h + 1,
                            "weight": 0.7,
                            "edge_type": "vertical",
                            "directed": False,
                        }
                    )

    Path(records_path).parent.mkdir(parents=True, exist_ok=True)
    Path(edges_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(records_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)
    if perception_path:
        Path(perception_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(perception_rows).to_csv(perception_path, index=False)


def generate_operational_sample(
    cells_path: str | Path,
    edges_path: str | Path,
    transmitters_path: str | Path,
    uavs_path: str | Path,
    tasks_path: str | Path,
) -> None:
    cells = []
    grid_rows = range(4)
    grid_cols = range(4)
    height_layers = range(3)
    spacing_m = 500.0
    for r in grid_rows:
        for c in grid_cols:
            for h in height_layers:
                grid_id = f"BDG-L18-R{r:02d}-C{c:02d}"
                corridor = c in (1, 2)
                cells.append(
                    {
                        "grid_id": grid_id,
                        "height_layer": h,
                        "center_x_m": c * spacing_m,
                        "center_y_m": r * spacing_m,
                        "center_z_m": 80 + h * 60,
                        "route_capacity": 8 + (4 if corridor else 0),
                        "occupancy": 0.15 + 0.05 * h,
                        "congestion_score": 0.25 + (0.25 if corridor and h == 1 else 0.0),
                        "em_interference": 0.15 + 0.08 * c,
                        "weather_wind": 4 + r,
                        "weather_visibility": 8000 - 300 * r,
                        "population_density": 2500 + 1200 * r + 600 * c,
                        "no_fly_flag": 1 if (r == 2 and c == 2 and h == 0) else 0,
                        "risk_score": 0.08 * r + 0.04 * c,
                        "height_ref": "AGL",
                    }
                )

    edge_rows = []
    for r in grid_rows:
        for c in grid_cols:
            for h in height_layers:
                source = f"BDG-L18-R{r:02d}-C{c:02d}"
                for dr, dc, edge_type in [(1, 0, "adjacent"), (0, 1, "route")]:
                    rr, cc = r + dr, c + dc
                    if rr in grid_rows and cc in grid_cols:
                        edge_rows.append(
                            {
                                "source_grid_id": source,
                                "source_height_layer": h,
                                "target_grid_id": f"BDG-L18-R{rr:02d}-C{cc:02d}",
                                "target_height_layer": h,
                                "weight": 1.0,
                                "edge_type": edge_type,
                                "directed": False,
                            }
                        )
                if h + 1 in height_layers:
                    edge_rows.append(
                        {
                            "source_grid_id": source,
                            "source_height_layer": h,
                            "target_grid_id": source,
                            "target_height_layer": h + 1,
                            "weight": 0.8,
                            "edge_type": "vertical",
                            "directed": False,
                        }
                    )

    transmitters = [
        {
            "transmitter_id": "BS-001",
            "x_m": -200,
            "y_m": -200,
            "z_m": 45,
            "frequency_mhz": 2400,
            "bandwidth_mhz": 20,
            "tx_power_dbm": 36,
            "tx_gain_dbi": 8,
            "rx_gain_dbi": 2,
            "noise_figure_db": 7,
            "role": "base_station",
        },
        {
            "transmitter_id": "RELAY-001",
            "x_m": 1200,
            "y_m": 900,
            "z_m": 160,
            "frequency_mhz": 2400,
            "bandwidth_mhz": 20,
            "tx_power_dbm": 30,
            "tx_gain_dbi": 5,
            "rx_gain_dbi": 2,
            "noise_figure_db": 7,
            "role": "relay",
        },
        {
            "transmitter_id": "JAM-001",
            "x_m": 1600,
            "y_m": 1400,
            "z_m": 80,
            "frequency_mhz": 2400,
            "bandwidth_mhz": 20,
            "tx_power_dbm": 18,
            "tx_gain_dbi": 0,
            "rx_gain_dbi": 0,
            "noise_figure_db": 7,
            "role": "jammer",
        },
    ]
    uavs = [
        {
            "uav_id": "UAV-001",
            "grid_id": "BDG-L18-R00-C00",
            "height_layer": 1,
            "x_m": 0,
            "y_m": 0,
            "z_m": 140,
            "vx_mps": 8,
            "vy_mps": 4,
            "vz_mps": 0,
            "speed_mps": 14,
            "battery_pct": 92,
            "max_range_m": 15000,
            "payload_capacity_kg": 3.0,
            "current_payload_kg": 0.4,
            "priority": 2,
        },
        {
            "uav_id": "UAV-002",
            "grid_id": "BDG-L18-R03-C00",
            "height_layer": 1,
            "x_m": 0,
            "y_m": 1500,
            "z_m": 140,
            "vx_mps": 8,
            "vy_mps": -5,
            "vz_mps": 0,
            "speed_mps": 13,
            "battery_pct": 78,
            "max_range_m": 12000,
            "payload_capacity_kg": 2.0,
            "current_payload_kg": 0.2,
            "priority": 1,
        },
    ]
    tasks = [
        {
            "task_id": "TASK-001",
            "origin_grid_id": "BDG-L18-R00-C01",
            "origin_height_layer": 1,
            "dest_grid_id": "BDG-L18-R03-C03",
            "dest_height_layer": 1,
            "priority": 5,
            "required_payload_kg": 1.0,
            "earliest_time": "2026-05-14T08:00:00",
            "latest_time": "2026-05-14T08:30:00",
        },
        {
            "task_id": "TASK-002",
            "origin_grid_id": "BDG-L18-R03-C00",
            "origin_height_layer": 1,
            "dest_grid_id": "BDG-L18-R00-C03",
            "dest_height_layer": 2,
            "priority": 3,
            "required_payload_kg": 0.5,
            "earliest_time": "2026-05-14T08:05:00",
            "latest_time": "2026-05-14T08:45:00",
        },
    ]

    for path in [cells_path, edges_path, transmitters_path, uavs_path, tasks_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cells).to_csv(cells_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)
    pd.DataFrame(transmitters).to_csv(transmitters_path, index=False)
    pd.DataFrame(uavs).to_csv(uavs_path, index=False)
    pd.DataFrame(tasks).to_csv(tasks_path, index=False)
