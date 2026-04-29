from __future__ import annotations

import math

import pandas as pd

from .routing import AStarRoutePlanner


class ResourceAllocationModel:
    """Baseline multi-UAV task and airspace allocation.

    Uses deterministic constrained greedy assignment. It is intentionally
    dependency-light; an OR-Tools CP-SAT implementation can later replace
    `allocate` with the same inputs and outputs.
    """

    def __init__(self, reserve_capacity: bool = True) -> None:
        self.reserve_capacity = reserve_capacity

    def allocate(self, cells: pd.DataFrame, edges: pd.DataFrame, uavs: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
        planner = AStarRoutePlanner(cells, edges)
        remaining_range = {
            row.uav_id: float(row.max_range_m) * max(float(row.battery_pct), 0.0) / 100.0
            for row in uavs.itertuples(index=False)
        }
        remaining_payload = {
            row.uav_id: max(0.0, float(row.payload_capacity_kg) - float(row.current_payload_kg))
            for row in uavs.itertuples(index=False)
        }
        reserved: dict[str, int] = {}
        assignments = []
        sorted_tasks = tasks.sort_values(["priority", "task_id"], ascending=[False, True])

        for task in sorted_tasks.itertuples(index=False):
            best = None
            for uav in uavs.itertuples(index=False):
                if remaining_payload[uav.uav_id] < float(task.required_payload_kg):
                    continue
                to_origin = planner.plan(uav.grid_id, int(uav.height_layer), task.origin_grid_id, int(task.origin_height_layer))
                mission = planner.plan(task.origin_grid_id, int(task.origin_height_layer), task.dest_grid_id, int(task.dest_height_layer))
                if not to_origin.found or not mission.found:
                    continue
                total_distance = to_origin.distance_m + mission.distance_m
                if total_distance > remaining_range[uav.uav_id]:
                    continue
                capacity_penalty = _capacity_penalty(mission.route, cells, reserved)
                score = total_distance + mission.risk_cost + capacity_penalty - float(task.priority) * 100.0
                if best is None or score < best["score"]:
                    best = {
                        "uav": uav,
                        "to_origin": to_origin,
                        "mission": mission,
                        "total_distance": total_distance,
                        "score": score,
                        "capacity_penalty": capacity_penalty,
                    }

            if best is None:
                assignments.append(
                    {
                        "task_id": task.task_id,
                        "uav_id": None,
                        "status": "unassigned",
                        "reason": "no feasible uav or route",
                        "route_grid_sequence": [],
                        "distance_m": math.inf,
                        "allocation_score": math.inf,
                        "load_ratio": None,
                    }
                )
                continue

            uav_id = best["uav"].uav_id
            remaining_range[uav_id] -= best["total_distance"]
            remaining_payload[uav_id] -= float(task.required_payload_kg)
            if self.reserve_capacity:
                for node_id in best["mission"].route:
                    reserved[node_id] = reserved.get(node_id, 0) + 1
            load_ratio = _route_load_ratio(best["mission"].route, cells, reserved)
            assignments.append(
                {
                    "task_id": task.task_id,
                    "uav_id": uav_id,
                    "status": "assigned",
                    "reason": "ok",
                    "route_grid_sequence": best["mission"].route,
                    "distance_m": round(best["total_distance"], 6),
                    "allocation_score": round(best["score"], 6),
                    "load_ratio": round(load_ratio, 6),
                }
            )
        return pd.DataFrame(assignments)


def _capacity_penalty(route: list[str], cells: pd.DataFrame, reserved: dict[str, int]) -> float:
    indexed = _capacity_index(cells)
    penalty = 0.0
    for node_id in route:
        capacity = max(indexed.get(node_id, 1.0), 1.0)
        load = reserved.get(node_id, 0) + 1
        if load > capacity:
            penalty += 10_000.0 * (load - capacity)
        else:
            penalty += 50.0 * (load / capacity)
    return penalty


def _route_load_ratio(route: list[str], cells: pd.DataFrame, reserved: dict[str, int]) -> float:
    indexed = _capacity_index(cells)
    ratios = [reserved.get(node_id, 0) / max(indexed.get(node_id, 1.0), 1.0) for node_id in route]
    return max(ratios) if ratios else 0.0


def _capacity_index(cells: pd.DataFrame) -> dict[str, float]:
    from .grid import make_node_id

    return {
        make_node_id(row.grid_id, int(row.height_layer)): float(getattr(row, "route_capacity", 1.0))
        for row in cells.itertuples(index=False)
    }

