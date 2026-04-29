from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FeatureConfig, build_prediction_features, build_supervised


@dataclass(slots=True)
class SpatialTemporalRidgeModel:
    """Tabular spatio-temporal baseline using lags, neighbor means, and exogenous features."""

    config: FeatureConfig
    pipeline: Pipeline
    feature_names: list[str]
    target_mean: float
    target_std: float
    train_samples: int

    @classmethod
    def fit(
        cls,
        records: pd.DataFrame,
        edges: pd.DataFrame,
        config: FeatureConfig,
        alpha: float = 1.0,
    ) -> "SpatialTemporalRidgeModel":
        data = build_supervised(records, edges, config)
        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=42)),
            ]
        )
        pipeline.fit(data.x, data.y)
        target_values = data.y.reshape(-1)
        return cls(
            config=config,
            pipeline=pipeline,
            feature_names=data.feature_names,
            target_mean=float(np.mean(target_values)),
            target_std=float(np.std(target_values) or 1.0),
            train_samples=int(data.x.shape[0]),
        )

    def predict(self, records: pd.DataFrame, edges: pd.DataFrame, horizon_steps: int | None = None):
        if horizon_steps is None:
            horizon_steps = self.config.horizon_steps
        if horizon_steps > self.config.horizon_steps:
            raise ValueError(
                f"Requested horizon_steps={horizon_steps}, model supports {self.config.horizon_steps}"
            )
        x, node_ids, anchor_time = build_prediction_features(
            records=records,
            edges=edges,
            config=self.config,
            feature_names=self.feature_names,
        )
        y = np.asarray(self.pipeline.predict(x), dtype=float)[:, :horizon_steps]
        return np.maximum(y, 0.0), node_ids, pd.Timestamp(anchor_time)

