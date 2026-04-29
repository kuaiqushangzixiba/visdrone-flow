from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(records_path: str | Path, edges_path: str | Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rows = []
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

