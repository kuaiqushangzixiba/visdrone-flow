from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelBundle:
    model_name: str
    model: Any
    trained_at: str
    metadata: dict[str, Any]


def save_bundle(path: str | Path, bundle: ModelBundle) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as file:
        pickle.dump(bundle, file)


def load_bundle(path: str | Path) -> ModelBundle:
    with Path(path).open("rb") as file:
        bundle = pickle.load(file)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"Invalid model bundle in {path}")
    return bundle


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

