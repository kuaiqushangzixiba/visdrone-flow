from __future__ import annotations

from pathlib import Path

from visdrone_flow.cli import _predict, _train
from visdrone_flow.io import read_json
from visdrone_flow.sample_data import generate_sample


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_train_and_predict(tmp_path: Path) -> None:
    records = tmp_path / "records.csv"
    edges = tmp_path / "edges.csv"
    artifact = tmp_path / "model.pkl"
    output = tmp_path / "predictions.json"
    generate_sample(records, edges)

    _train(
        Args(
            records=str(records),
            edges=str(edges),
            artifact=str(artifact),
            model="ridge",
            history_steps=12,
            horizon_steps=6,
            target_column="flow_in",
            alpha=1.0,
        )
    )
    assert artifact.exists()

    _predict(
        Args(
            artifact=str(artifact),
            records=str(records),
            edges=str(edges),
            output=str(output),
            horizon_steps=6,
            time_granularity_minutes=5,
        )
    )
    payload = read_json(output)
    assert payload["model_name"] == "SpatialTemporalRidge"
    assert payload["horizon_steps"] == 6
    assert len(payload["predictions"]) > 0
    first = payload["predictions"][0]
    assert first["grid_id"]
    assert first["pred_flow"] >= 0

