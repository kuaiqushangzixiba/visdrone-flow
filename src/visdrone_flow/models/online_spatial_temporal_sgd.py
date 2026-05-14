from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from ..features import FeatureConfig, build_prediction_features, build_supervised


@dataclass(slots=True)
class OnlineSpatialTemporalSGDModel:
    """Online spatio-temporal forecaster using incremental SGD regressors.

    It keeps the same feature contract as SpatialTemporalRidgeModel, but supports
    `partial_fit` for streaming flight/perception updates.
    """

    config: FeatureConfig
    scaler: StandardScaler
    regressors: list[SGDRegressor]
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
        alpha: float = 0.0001,
    ) -> "OnlineSpatialTemporalSGDModel":
        data = build_supervised(records, edges, config)
        scaler = StandardScaler()
        scaler.partial_fit(data.x)
        x_scaled = scaler.transform(data.x)
        regressors = [
            SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=alpha,
                learning_rate="invscaling",
                eta0=0.01,
                random_state=42 + horizon,
                max_iter=2000,
                tol=1e-4,
            )
            for horizon in range(config.horizon_steps)
        ]
        for horizon, regressor in enumerate(regressors):
            regressor.partial_fit(x_scaled, data.y[:, horizon])
        target_values = data.y.reshape(-1)
        return cls(
            config=config,
            scaler=scaler,
            regressors=regressors,
            feature_names=data.feature_names,
            target_mean=float(np.mean(target_values)),
            target_std=float(np.std(target_values) or 1.0),
            train_samples=int(data.x.shape[0]),
        )

    def partial_fit(self, records: pd.DataFrame, edges: pd.DataFrame) -> int:
        data = build_supervised(records, edges, self.config)
        if data.feature_names != self.feature_names:
            raise ValueError("Feature config does not match trained model")
        self.scaler.partial_fit(data.x)
        x_scaled = self.scaler.transform(data.x)
        for horizon, regressor in enumerate(self.regressors):
            regressor.partial_fit(x_scaled, data.y[:, horizon])
        target_values = data.y.reshape(-1)
        total = self.train_samples + int(data.x.shape[0])
        self.target_mean = (self.target_mean * self.train_samples + float(np.sum(target_values))) / total
        self.target_std = float(max(np.std(target_values), self.target_std * 0.8, 1e-6))
        self.train_samples = total
        return int(data.x.shape[0])

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
        x_scaled = self.scaler.transform(x)
        columns = [regressor.predict(x_scaled) for regressor in self.regressors[:horizon_steps]]
        y = np.vstack(columns).T
        return np.maximum(y, 0.0), node_ids, pd.Timestamp(anchor_time)

