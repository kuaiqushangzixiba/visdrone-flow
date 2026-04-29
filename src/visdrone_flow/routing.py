from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import pandas as pd

from .grid import make_node_id, split_node_id


@dataclass(slots=True)
class RouteResult:
    found: bool
    route: list[str]
    total_cost: float
    distance_m: float
    risk_cost: float
    message: str = ""


class AStarRoutePlanner:
    """3D grid A* route planner with congestion, EM, weather, and risk costs."""

    def __init__(
        self,
        cells: pd.DataFrame,
        edges: pd.DataFrame,
        distance_weight: float = 1.0,
        congestion_weight: float = 120.0,
        em_weight: float = 60.0,
        risk_weight: float = 300.0,
        height_change_weight: float = 0.4,
    ) -> None:
        self.cells = _index_cells(cells)
        self.adj = _build_adjacency(edges)
        self.distance_weight = distance_weight
        self.congestion_weight = congestion_weight
        self.em_weight = em_weight
        self.risk_weight = risk_weight
        self.height_change_weight = height_change_weight

    def plan(
        self,
        start_grid_id: str,
        start_height_layer: int,
        end_grid_id: str,
        end_height_layer: int,
        max_nodes: int = 10_000,
    ) -> RouteResult:
        start = make_node_id(start_grid_id, start_height_layer)
        goal = make_node_id(end_grid_id, end_height_layer)
        if start not in self.cells:
            return RouteResult(False, [], math.inf, 0.0, 0.0, f"unknown start node {start}")
        if goal not in self.cells:
            return RouteResult(False, [], math.inf, 0.0, 0.0, f"unknown goal node {goal}")

        open_heap: list[tuple[float, str]] = [(0.0, start)]
        came_from: dict[str, str] = {}
        g_score = {start: 0.0}
        distance_score = {start: 0.0}
        risk_score = {start: 0.0}
        visited = 0

        while open_heap and visited < max_nodes:
            _, current = heapq.heappop(open_heap)
            visited += 1
            if current == goal:
                route = _reconstruct(came_from, current)
                return RouteResult(
                    True,
                    route,
                    round(g_score[current], 6),
                    round(distance_score[current], 6),
                    round(risk_score[current], 6),
                    "ok",
                )
            for neighbor, edge_weight in self.adj.get(current, []):
                if neighbor not in self.cells:
                    continue
                step_distance = _distance(self.cells[current], self.cells[neighbor])
                step_risk = self._node_risk(neighbor)
                step_cost = (
                    self.distance_weight * step_distance * edge_weight
                    + self.height_change_weight * abs(self.cells[current]["center_z_m"] - self.cells[neighbor]["center_z_m"])
                    + step_risk
                )
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                distance_score[neighbor] = distance_score[current] + step_distance
                risk_score[neighbor] = risk_score[current] + step_risk
                priority = tentative + self._heuristic(neighbor, goal)
                heapq.heappush(open_heap, (priority, neighbor))
        return RouteResult(False, [], math.inf, 0.0, 0.0, "route not found")

    def _heuristic(self, node: str, goal: str) -> float:
        return self.distance_weight * _distance(self.cells[node], self.cells[goal])

    def _node_risk(self, node: str) -> float:
        cell = self.cells[node]
        if int(cell.get("no_fly_flag", 0)) == 1:
            return 1_000_000.0
        visibility = max(float(cell.get("weather_visibility", 10_000)), 1.0)
        visibility_penalty = max(0.0, (2000.0 - visibility) / 2000.0)
        wind_penalty = max(0.0, (float(cell.get("weather_wind", 0.0)) - 10.0) / 10.0)
        return (
            self.congestion_weight * float(cell.get("congestion_score", 0.0))
            + self.em_weight * float(cell.get("em_interference", 0.0))
            + self.risk_weight * float(cell.get("risk_score", 0.0))
            + self.risk_weight * visibility_penalty
            + self.risk_weight * wind_penalty
        )


def route_to_dataframe(result: RouteResult) -> pd.DataFrame:
    rows = []
    for order, node_id in enumerate(result.route):
        node = split_node_id(node_id)
        rows.append({"seq": order, "grid_id": node.grid_id, "height_layer": node.height_layer, "node_id": node_id})
    return pd.DataFrame(rows)


def _index_cells(cells: pd.DataFrame) -> dict[str, dict[str, float]]:
    indexed: dict[str, dict[str, float]] = {}
    for row in cells.to_dict(orient="records"):
        node_id = make_node_id(str(row["grid_id"]), int(row["height_layer"]))
        indexed[node_id] = row
    return indexed


def _build_adjacency(edges: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for row in edges.itertuples(index=False):
        source = make_node_id(row.source_grid_id, int(row.source_height_layer))
        target = make_node_id(row.target_grid_id, int(row.target_height_layer))
        weight = float(getattr(row, "weight", 1.0))
        adjacency.setdefault(source, []).append((target, weight))
        if not bool(getattr(row, "directed", False)):
            adjacency.setdefault(target, []).append((source, weight))
    return adjacency


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    dx = float(a["center_x_m"]) - float(b["center_x_m"])
    dy = float(a["center_y_m"]) - float(b["center_y_m"])
    dz = float(a["center_z_m"]) - float(b["center_z_m"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _reconstruct(came_from: dict[str, str], current: str) -> list[str]:
    route = [current]
    while current in came_from:
        current = came_from[current]
        route.append(current)
    route.reverse()
    return route

