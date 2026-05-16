from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .allocation import ResourceAllocationModel
from .api import run_server
from .electromagnetic import ElectromagneticEnvironmentModel
from .features import FeatureConfig
from .io import read_edges_csv, read_optional_csv, read_records_csv, write_json
from .model_store import ModelBundle, load_bundle, save_bundle, utc_now_iso
from .models import HistoricalAverageModel, OnlineSpatialTemporalSGDModel, SpatialTemporalRidgeModel
from .perception import build_standard_flow_dataset
from .routing import AStarRoutePlanner, route_to_dataframe
from .safety import SafetyAssessmentModel
from .sample_data import generate_operational_sample, generate_sample
from .schemas import FlowRecord, PredictionRequest
from .service import FlowForecastService
from .state_io import read_cells_csv, read_tasks_csv, read_transmitters_csv, read_uavs_csv


def main() -> None:
    parser = argparse.ArgumentParser(prog="visdrone-flow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("generate-sample")
    sample.add_argument("--records", default="examples/sample_flow.csv")
    sample.add_argument("--edges", default="examples/sample_edges.csv")
    sample.add_argument("--perception", default="examples/sample_perception.csv")

    operational = subparsers.add_parser("generate-operational-sample")
    operational.add_argument("--cells", default="examples/sample_cells.csv")
    operational.add_argument("--edges", default="examples/sample_operational_edges.csv")
    operational.add_argument("--transmitters", default="examples/sample_transmitters.csv")
    operational.add_argument("--uavs", default="examples/sample_uavs.csv")
    operational.add_argument("--tasks", default="examples/sample_tasks.csv")

    train = subparsers.add_parser("train")
    train.add_argument("--records", required=True)
    train.add_argument("--edges", default=None)
    train.add_argument("--perception", default=None)
    train.add_argument("--artifact", required=True)
    train.add_argument("--model", choices=["ridge", "historical", "online"], default="ridge")
    train.add_argument("--history-steps", type=int, default=12)
    train.add_argument("--horizon-steps", type=int, default=6)
    train.add_argument("--target-column", default="flow_in")
    train.add_argument("--alpha", type=float, default=1.0)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--records", required=True)
    predict.add_argument("--edges", default=None)
    predict.add_argument("--perception", default=None)
    predict.add_argument("--output", required=True)
    predict.add_argument("--horizon-steps", type=int, default=6)
    predict.add_argument("--time-granularity-minutes", type=int, default=5)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--artifact", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)

    em = subparsers.add_parser("analyze-em")
    em.add_argument("--cells", required=True)
    em.add_argument("--transmitters", required=True)
    em.add_argument("--output", required=True)

    allocation = subparsers.add_parser("allocate-resources")
    allocation.add_argument("--cells", required=True)
    allocation.add_argument("--edges", required=True)
    allocation.add_argument("--uavs", required=True)
    allocation.add_argument("--tasks", required=True)
    allocation.add_argument("--output", required=True)
    allocation.add_argument("--time-granularity-minutes", type=int, default=5)
    allocation.add_argument("--safety-buffer-slots", type=int, default=1)

    route = subparsers.add_parser("plan-route")
    route.add_argument("--cells", required=True)
    route.add_argument("--edges", required=True)
    route.add_argument("--start-grid", required=True)
    route.add_argument("--start-height", type=int, required=True)
    route.add_argument("--end-grid", required=True)
    route.add_argument("--end-height", type=int, required=True)
    route.add_argument("--output", required=True)

    safety = subparsers.add_parser("assess-safety")
    safety.add_argument("--cells", required=True)
    safety.add_argument("--uavs", required=True)
    safety.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "generate-sample":
        generate_sample(args.records, args.edges, perception_path=args.perception)
        print(f"generated records={args.records} edges={args.edges} perception={args.perception}")
    elif args.command == "generate-operational-sample":
        generate_operational_sample(args.cells, args.edges, args.transmitters, args.uavs, args.tasks)
        print(
            "generated "
            f"cells={args.cells} edges={args.edges} transmitters={args.transmitters} "
            f"uavs={args.uavs} tasks={args.tasks}"
        )
    elif args.command == "train":
        _train(args)
    elif args.command == "predict":
        _predict(args)
    elif args.command == "serve":
        run_server(args.artifact, args.host, args.port)
    elif args.command == "analyze-em":
        _analyze_em(args)
    elif args.command == "allocate-resources":
        _allocate_resources(args)
    elif args.command == "plan-route":
        _plan_route(args)
    elif args.command == "assess-safety":
        _assess_safety(args)


def _train(args: argparse.Namespace) -> None:
    records = read_records_csv(args.records)
    perception = read_optional_csv(args.perception)
    records = build_standard_flow_dataset(records, perception)
    edges = read_edges_csv(args.edges)
    config = FeatureConfig(
        target_column=args.target_column,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
    )
    if args.model == "historical":
        model = HistoricalAverageModel.fit(records, config)
        name = "HistoricalAverage"
    elif args.model == "online":
        model = OnlineSpatialTemporalSGDModel.fit(records, edges, config, alpha=args.alpha)
        name = "OnlineSpatialTemporalSGD"
    else:
        model = SpatialTemporalRidgeModel.fit(records, edges, config, alpha=args.alpha)
        name = "SpatialTemporalRidge"
    bundle = ModelBundle(
        model_name=name,
        model=model,
        trained_at=utc_now_iso(),
        metadata={
            "records": str(Path(args.records)),
            "edges": str(Path(args.edges)) if args.edges else None,
            "perception": str(Path(args.perception)) if args.perception else None,
            "history_steps": args.history_steps,
            "horizon_steps": args.horizon_steps,
            "target_column": args.target_column,
        },
    )
    save_bundle(args.artifact, bundle)
    print(f"saved {name} model to {args.artifact}")


def _predict(args: argparse.Namespace) -> None:
    records = read_records_csv(args.records)
    perception = read_optional_csv(args.perception)
    records = build_standard_flow_dataset(records, perception)
    edges = read_edges_csv(args.edges)
    request = PredictionRequest.model_validate(
        {
            "request_id": "cli",
            "horizon_steps": args.horizon_steps,
            "time_granularity_minutes": args.time_granularity_minutes,
            "records": json.loads(
                records[[column for column in FlowRecord.model_fields if column in records.columns]].to_json(
                    orient="records",
                    date_format="iso",
                )
            ),
            "edges": json.loads(edges.to_json(orient="records")) if not edges.empty else [],
        }
    )
    service = FlowForecastService(load_bundle(args.artifact))
    response = service.predict(request)
    write_json(args.output, response.model_dump(mode="json"))
    print(f"wrote predictions to {args.output}")


def _analyze_em(args: argparse.Namespace) -> None:
    cells = read_cells_csv(args.cells)
    transmitters = read_transmitters_csv(args.transmitters)
    result = ElectromagneticEnvironmentModel().analyze(cells, transmitters)
    _write_frame(args.output, result)
    print(f"wrote electromagnetic analysis to {args.output}")


def _allocate_resources(args: argparse.Namespace) -> None:
    cells = read_cells_csv(args.cells)
    edges = read_edges_csv(args.edges)
    uavs = read_uavs_csv(args.uavs)
    tasks = read_tasks_csv(args.tasks)
    from .allocation import ReservationPolicy

    result = ResourceAllocationModel(
        policy=ReservationPolicy(
            time_granularity_minutes=args.time_granularity_minutes,
            safety_buffer_slots=args.safety_buffer_slots,
        )
    ).allocate(cells, edges, uavs, tasks)
    _write_frame(args.output, result)
    print(f"wrote resource allocation to {args.output}")


def _plan_route(args: argparse.Namespace) -> None:
    cells = read_cells_csv(args.cells)
    edges = read_edges_csv(args.edges)
    planner = AStarRoutePlanner(cells, edges)
    result = planner.plan(args.start_grid, args.start_height, args.end_grid, args.end_height)
    payload = {
        "found": result.found,
        "total_cost": result.total_cost,
        "distance_m": result.distance_m,
        "risk_cost": result.risk_cost,
        "message": result.message,
        "route": route_to_dataframe(result).to_dict(orient="records"),
    }
    write_json(args.output, payload)
    print(f"wrote route plan to {args.output}")


def _assess_safety(args: argparse.Namespace) -> None:
    cells = read_cells_csv(args.cells)
    uavs = read_uavs_csv(args.uavs)
    result = SafetyAssessmentModel().assess(cells, uavs)
    payload = {name: frame.to_dict(orient="records") for name, frame in result.items()}
    write_json(args.output, payload)
    print(f"wrote safety assessment to {args.output}")


def _write_frame(path: str, frame: pd.DataFrame) -> None:
    if Path(path).suffix.lower() == ".csv":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return
    write_json(path, frame.to_dict(orient="records"))


if __name__ == "__main__":
    main()
