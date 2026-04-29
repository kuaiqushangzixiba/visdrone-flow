from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import run_server
from .features import FeatureConfig
from .io import read_edges_csv, read_records_csv, write_json
from .model_store import ModelBundle, load_bundle, save_bundle, utc_now_iso
from .models import HistoricalAverageModel, SpatialTemporalRidgeModel
from .sample_data import generate_sample
from .schemas import PredictionRequest
from .service import FlowForecastService


def main() -> None:
    parser = argparse.ArgumentParser(prog="visdrone-flow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("generate-sample")
    sample.add_argument("--records", default="examples/sample_flow.csv")
    sample.add_argument("--edges", default="examples/sample_edges.csv")

    train = subparsers.add_parser("train")
    train.add_argument("--records", required=True)
    train.add_argument("--edges", default=None)
    train.add_argument("--artifact", required=True)
    train.add_argument("--model", choices=["ridge", "historical"], default="ridge")
    train.add_argument("--history-steps", type=int, default=12)
    train.add_argument("--horizon-steps", type=int, default=6)
    train.add_argument("--target-column", default="flow_in")
    train.add_argument("--alpha", type=float, default=1.0)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--records", required=True)
    predict.add_argument("--edges", default=None)
    predict.add_argument("--output", required=True)
    predict.add_argument("--horizon-steps", type=int, default=6)
    predict.add_argument("--time-granularity-minutes", type=int, default=5)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--artifact", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)

    args = parser.parse_args()
    if args.command == "generate-sample":
        generate_sample(args.records, args.edges)
        print(f"generated records={args.records} edges={args.edges}")
    elif args.command == "train":
        _train(args)
    elif args.command == "predict":
        _predict(args)
    elif args.command == "serve":
        run_server(args.artifact, args.host, args.port)


def _train(args: argparse.Namespace) -> None:
    records = read_records_csv(args.records)
    edges = read_edges_csv(args.edges)
    config = FeatureConfig(
        target_column=args.target_column,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
    )
    if args.model == "historical":
        model = HistoricalAverageModel.fit(records, config)
        name = "HistoricalAverage"
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
            "history_steps": args.history_steps,
            "horizon_steps": args.horizon_steps,
            "target_column": args.target_column,
        },
    )
    save_bundle(args.artifact, bundle)
    print(f"saved {name} model to {args.artifact}")


def _predict(args: argparse.Namespace) -> None:
    records = read_records_csv(args.records)
    edges = read_edges_csv(args.edges)
    request = PredictionRequest.model_validate(
        {
            "request_id": "cli",
            "horizon_steps": args.horizon_steps,
            "time_granularity_minutes": args.time_granularity_minutes,
            "records": json.loads(records.to_json(orient="records", date_format="iso")),
            "edges": json.loads(edges.to_json(orient="records")) if not edges.empty else [],
        }
    )
    service = FlowForecastService(load_bundle(args.artifact))
    response = service.predict(request)
    write_json(args.output, response.model_dump(mode="json"))
    print(f"wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

