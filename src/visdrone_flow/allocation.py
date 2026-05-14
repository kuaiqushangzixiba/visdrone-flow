from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .routing import AStarRoutePlanner


@dataclass(slots=True)
class ReservationPolicy:
    time_granularity_minutes: int = 5
    safety_buffer_slots: int = 1
    balance_weight: float = 350.0
    capacity_weight: float = 10_000.0


class ResourceAllocationModel:
    """Baseline multi-UAV task and airspace allocation.

    Uses deterministic constrained greedy assignment. It is intentionally
    dependency-light; an OR-Tools CP-SAT implementation can later replace
    `allocate` with the same inputs and outputs.
    """

    def __init__(self, reserve_capacity: bool = True, policy: ReservationPolicy | None = None) -> None:
        self.reserve_capacity = reserve_capacity
        self.policy = policy or ReservationPolicy()

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
        assigned_distance = {row.uav_id: 0.0 for row in uavs.itertuples(index=False)}
        assigned_tasks = {row.uav_id: 0 for row in uavs.itertuples(index=False)}
        reservations: dict[tuple[str, int], int] = {}
        assignments = []
        sorted_tasks = tasks.sort_values(["priority", "task_id"], ascending=[False, True])
        base_time = _base_time(sorted_tasks)

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
                start_time = _task_start_time(task, base_time)
                route_schedule = _build_route_schedule(
                    mission.route,
                    cells,
                    speed_mps=float(getattr(uav, "speed_mps", 12.0)),
                    start_time=start_time,
                    time_granularity_minutes=self.policy.time_granularity_minutes,
                )
                capacity_penalty = _capacity_penalty(
                    route_schedule,
                    cells,
                    reservations,
                    self.policy.safety_buffer_slots,
                    self.policy.capacity_weight,
                )
                balance_penalty = (
                    assigned_tasks[uav.uav_id] * self.policy.balance_weight
                    + assigned_distance[uav.uav_id] * 0.05
                )
                score = (
                    total_distance
                    + mission.risk_cost
                    + capacity_penalty
                    + balance_penalty
                    - float(task.priority) * 100.0
                )
                if best is None or score < best["score"]:
                    best = {
                        "uav": uav,
                        "to_origin": to_origin,
                        "mission": mission,
                        "total_distance": total_distance,
                        "score": score,
                        "capacity_penalty": capacity_penalty,
                        "balance_penalty": balance_penalty,
                        "route_schedule": route_schedule,
                    }

            if best is None:
                assignments.append(
                    {
                        "task_id": task.task_id,
                        "uav_id": None,
                        "status": "unassigned",
                        "reason": "no feasible uav or route",
                        "route_grid_sequence": [],
                        "flight_plan": [],
                        "distance_m": math.inf,
                        "allocation_score": math.inf,
                        "load_ratio": None,
                        "task_execution_advice": "relax time window, add UAV, or expand available airspace",
                    }
                )
                continue

            uav_id = best["uav"].uav_id
            remaining_range[uav_id] -= best["total_distance"]
            remaining_payload[uav_id] -= float(task.required_payload_kg)
            assigned_distance[uav_id] += best["total_distance"]
            assigned_tasks[uav_id] += 1
            if self.reserve_capacity:
                _reserve(best["route_schedule"], reservations, self.policy.safety_buffer_slots)
            load_ratio = _route_load_ratio(best["route_schedule"], cells, reservations)
            assignments.append(
                {
                    "task_id": task.task_id,
                    "uav_id": uav_id,
                    "status": "assigned",
                    "reason": "ok",
                    "route_grid_sequence": best["mission"].route,
                    "flight_plan": best["route_schedule"],
                    "distance_m": round(best["total_distance"], 6),
                    "allocation_score": round(best["score"], 6),
                    "capacity_penalty": round(best["capacity_penalty"], 6),
                    "balance_penalty": round(best["balance_penalty"], 6),
                    "load_ratio": round(load_ratio, 6),
                    "task_execution_advice": _task_advice(load_ratio, best["mission"].risk_cost),
                }
            )
        return pd.DataFrame(assignments)


def _capacity_penalty(
    route_schedule: list[dict[str, object]],
    cells: pd.DataFrame,
    reservations: dict[tuple[str, int], int],
    safety_buffer_slots: int,
    capacity_weight: float,
) -> float:
    indexed = _capacity_index(cells)
    penalty = 0.0
    for step in route_schedule:
        node_id = str(step["node_id"])
        slot = int(step["slot"])
        capacity = max(indexed.get(node_id, 1.0), 1.0)
        load = 1 + max(
            reservations.get((node_id, buffered_slot), 0)
            for buffered_slot in range(slot - safety_buffer_slots, slot + safety_buffer_slots + 1)
        )
        if load > capacity:
            penalty += capacity_weight * (load - capacity)
        else:
            penalty += 50.0 * (load / capacity)
    return penalty


def _route_load_ratio(
    route_schedule: list[dict[str, object]],
    cells: pd.DataFrame,
    reservations: dict[tuple[str, int], int],
) -> float:
    indexed = _capacity_index(cells)
    ratios = [
        reservations.get((str(step["node_id"]), int(step["slot"])), 0)
        / max(indexed.get(str(step["node_id"]), 1.0), 1.0)
        for step in route_schedule
    ]
    return max(ratios) if ratios else 0.0


def _capacity_index(cells: pd.DataFrame) -> dict[str, float]:
    from .grid import make_node_id

    return {
        make_node_id(row.grid_id, int(row.height_layer)): float(getattr(row, "route_capacity", 1.0))
        for row in cells.itertuples(index=False)
    }


def _base_time(tasks: pd.DataFrame) -> pd.Timestamp:
    for column in ["earliest_time", "latest_time"]:
        if column in tasks.columns:
            values = pd.to_datetime(tasks[column], errors="coerce").dropna()
            if not values.empty:
                return pd.Timestamp(values.min())
    return pd.Timestamp("2026-05-14T08:00:00")


def _task_start_time(task: object, fallback: pd.Timestamp) -> pd.Timestamp:
    value = getattr(task, "earliest_time", None)
    if value is None or pd.isna(value):
        return fallback
    return pd.Timestamp(value)


def _build_route_schedule(
    route: list[str],
    cells: pd.DataFrame,
    speed_mps: float,
    start_time: pd.Timestamp,
    time_granularity_minutes: int,
) -> list[dict[str, object]]:
    from .routing import _distance, _index_cells

    indexed = _index_cells(cells)
    current_time = pd.Timestamp(start_time)
    result = []
    previous_node: str | None = None
    for seq, node_id in enumerate(route):
        if previous_node is not None:
            distance_m = _distance(indexed[previous_node], indexed[node_id])
            current_time += pd.Timedelta(seconds=distance_m / max(speed_mps, 1e-6))
        slot = int((current_time - start_time).total_seconds() // (time_granularity_minutes * 60))
        result.append(
            {
                "seq": seq,
                "node_id": node_id,
                "eta": current_time.isoformat(),
                "slot": slot,
            }
        )
        previous_node = node_id
    return result


def _reserve(route_schedule: list[dict[str, object]], reservations: dict[tuple[str, int], int], safety_buffer_slots: int) -> None:
    for step in route_schedule:
        node_id = str(step["node_id"])
        slot = int(step["slot"])
        for buffered_slot in range(slot - safety_buffer_slots, slot + safety_buffer_slots + 1):
            reservations[(node_id, buffered_slot)] = reservations.get((node_id, buffered_slot), 0) + 1


def _task_advice(load_ratio: float, risk_cost: float) -> str:
    if load_ratio >= 0.9:
        return "delay_or_raise_altitude_to_release_capacity"
    if risk_cost >= 1000:
        return "reroute_to_lower_risk_corridor"
    if load_ratio >= 0.65:
        return "monitor_capacity_and_keep_backup_route"
    return "execute_as_planned"

