from __future__ import annotations

from pathlib import Path

from visdrone_flow.allocation import ResourceAllocationModel
from visdrone_flow.electromagnetic import ElectromagneticEnvironmentModel
from visdrone_flow.io import read_edges_csv
from visdrone_flow.routing import AStarRoutePlanner
from visdrone_flow.safety import SafetyAssessmentModel
from visdrone_flow.sample_data import generate_operational_sample
from visdrone_flow.state_io import read_cells_csv, read_tasks_csv, read_transmitters_csv, read_uavs_csv


def _sample(tmp_path: Path):
    cells = tmp_path / "cells.csv"
    edges = tmp_path / "edges.csv"
    transmitters = tmp_path / "transmitters.csv"
    uavs = tmp_path / "uavs.csv"
    tasks = tmp_path / "tasks.csv"
    generate_operational_sample(cells, edges, transmitters, uavs, tasks)
    return cells, edges, transmitters, uavs, tasks


def test_electromagnetic_model(tmp_path: Path) -> None:
    cells_path, _, transmitters_path, _, _ = _sample(tmp_path)
    cells = read_cells_csv(cells_path)
    transmitters = read_transmitters_csv(transmitters_path)
    result = ElectromagneticEnvironmentModel().analyze(cells, transmitters)
    assert len(result) == len(cells)
    assert {"rssi_dbm", "sinr_db", "communication_stability", "avoid_flag"} <= set(result.columns)
    assert result["communication_stability"].between(0, 1).all()


def test_route_allocation_and_safety(tmp_path: Path) -> None:
    cells_path, edges_path, _, uavs_path, tasks_path = _sample(tmp_path)
    cells = read_cells_csv(cells_path)
    edges = read_edges_csv(edges_path)
    uavs = read_uavs_csv(uavs_path)
    tasks = read_tasks_csv(tasks_path)

    route = AStarRoutePlanner(cells, edges).plan("BDG-L18-R00-C00", 1, "BDG-L18-R03-C03", 1)
    assert route.found
    assert route.route[0] == "BDG-L18-R00-C00#H1"
    assert route.route[-1] == "BDG-L18-R03-C03#H1"

    allocation = ResourceAllocationModel().allocate(cells, edges, uavs, tasks)
    assert set(allocation["status"]) == {"assigned"}
    assert allocation["uav_id"].notna().all()
    assert allocation["flight_plan"].map(len).gt(0).all()
    assert {"capacity_penalty", "balance_penalty", "task_execution_advice"} <= set(allocation.columns)

    safety = SafetyAssessmentModel().assess(cells, uavs)
    assert {"summary", "conflicts", "grid_risks"} <= set(safety)
    assert safety["summary"].iloc[0]["overall_risk_score"] >= 0

