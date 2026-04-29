from __future__ import annotations

import math

import pandas as pd


class SafetyAssessmentModel:
    """Low-altitude safety assessment with CPA/TCPA and grid risk matrix."""

    def __init__(
        self,
        horizontal_separation_m: float = 120.0,
        vertical_separation_m: float = 30.0,
        lookahead_s: float = 300.0,
    ) -> None:
        self.horizontal_separation_m = horizontal_separation_m
        self.vertical_separation_m = vertical_separation_m
        self.lookahead_s = lookahead_s

    def assess(self, cells: pd.DataFrame, uavs: pd.DataFrame) -> dict[str, pd.DataFrame]:
        conflict_df = self.detect_conflicts(uavs)
        grid_df = self.grid_risk(cells)
        total_conflict = 0.0 if conflict_df.empty else float(conflict_df["conflict_risk"].max())
        grid_max = 0.0 if grid_df.empty else float(grid_df["grid_risk_score"].max())
        overall = min(1.0, 0.55 * grid_max + 0.45 * total_conflict)
        summary = pd.DataFrame(
            [
                {
                    "overall_risk_score": round(overall, 6),
                    "risk_level": _risk_level(overall),
                    "recommended_action": _recommended_action(overall),
                    "conflict_count": int((conflict_df.get("warning_level", pd.Series(dtype=int)) >= 2).sum()),
                    "unsafe_grid_count": int((grid_df.get("warning_level", pd.Series(dtype=int)) >= 2).sum()),
                }
            ]
        )
        return {"summary": summary, "conflicts": conflict_df, "grid_risks": grid_df}

    def detect_conflicts(self, uavs: pd.DataFrame) -> pd.DataFrame:
        rows = []
        records = list(uavs.itertuples(index=False))
        for i, own in enumerate(records):
            for intruder in records[i + 1 :]:
                cpa = _cpa(own, intruder, self.lookahead_s)
                horizontal = cpa["horizontal_m"]
                vertical = cpa["vertical_m"]
                h_ratio = max(0.0, 1.0 - horizontal / self.horizontal_separation_m)
                v_ratio = max(0.0, 1.0 - vertical / self.vertical_separation_m)
                time_ratio = max(0.0, 1.0 - cpa["tcpa_s"] / self.lookahead_s)
                risk = min(1.0, 0.45 * h_ratio + 0.35 * v_ratio + 0.20 * time_ratio)
                rows.append(
                    {
                        "ownship_id": own.uav_id,
                        "intruder_id": intruder.uav_id,
                        "tcpa_s": round(cpa["tcpa_s"], 6),
                        "horizontal_cpa_m": round(horizontal, 6),
                        "vertical_cpa_m": round(vertical, 6),
                        "conflict_risk": round(risk, 6),
                        "warning_level": _warning_level(risk),
                        "recommended_action": "separate_altitude_or_delay" if risk >= 0.65 else "monitor",
                    }
                )
        return pd.DataFrame(rows)

    def grid_risk(self, cells: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for cell in cells.itertuples(index=False):
            no_fly = 1.0 if int(getattr(cell, "no_fly_flag", 0)) == 1 else 0.0
            wind = max(0.0, (float(getattr(cell, "weather_wind", 0.0)) - 10.0) / 12.0)
            visibility = max(0.0, (2000.0 - float(getattr(cell, "weather_visibility", 10_000.0))) / 2000.0)
            population = min(1.0, float(getattr(cell, "population_density", 0.0)) / 20_000.0)
            congestion = min(1.0, float(getattr(cell, "congestion_score", 0.0)))
            em = min(1.0, float(getattr(cell, "em_interference", 0.0)))
            inherited = min(1.0, float(getattr(cell, "risk_score", 0.0)))
            risk = max(
                no_fly,
                min(1.0, 0.2 * wind + 0.2 * visibility + 0.2 * population + 0.15 * congestion + 0.15 * em + 0.1 * inherited),
            )
            rows.append(
                {
                    "grid_id": cell.grid_id,
                    "height_layer": int(cell.height_layer),
                    "grid_risk_score": round(risk, 6),
                    "warning_level": _warning_level(risk),
                    "risk_level": _risk_level(risk),
                    "recommended_action": _recommended_action(risk),
                }
            )
        return pd.DataFrame(rows)


def _cpa(a: object, b: object, lookahead_s: float) -> dict[str, float]:
    rx = float(a.x_m) - float(b.x_m)
    ry = float(a.y_m) - float(b.y_m)
    rz = float(a.z_m) - float(b.z_m)
    vx = float(getattr(a, "vx_mps", 0.0)) - float(getattr(b, "vx_mps", 0.0))
    vy = float(getattr(a, "vy_mps", 0.0)) - float(getattr(b, "vy_mps", 0.0))
    vz = float(getattr(a, "vz_mps", 0.0)) - float(getattr(b, "vz_mps", 0.0))
    vv = vx * vx + vy * vy + vz * vz
    tcpa = 0.0 if vv <= 1e-9 else -((rx * vx + ry * vy + rz * vz) / vv)
    tcpa = min(max(tcpa, 0.0), lookahead_s)
    cx = rx + vx * tcpa
    cy = ry + vy * tcpa
    cz = rz + vz * tcpa
    return {"tcpa_s": tcpa, "horizontal_m": math.sqrt(cx * cx + cy * cy), "vertical_m": abs(cz)}


def _warning_level(risk: float) -> int:
    if risk >= 0.85:
        return 3
    if risk >= 0.65:
        return 2
    if risk >= 0.35:
        return 1
    return 0


def _risk_level(risk: float) -> str:
    return ["normal", "attention", "high", "severe"][_warning_level(risk)]


def _recommended_action(risk: float) -> str:
    if risk >= 0.85:
        return "reject_or_reroute"
    if risk >= 0.65:
        return "manual_review_or_delay"
    if risk >= 0.35:
        return "monitor"
    return "allow"

