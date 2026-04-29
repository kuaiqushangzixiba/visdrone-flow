from __future__ import annotations

import math

import numpy as np
import pandas as pd


class ElectromagneticEnvironmentModel:
    """Baseline electromagnetic model using link budget, FSPL, and SINR.

    This is the deployable P0 model. A Sionna RT adapter can later replace
    `_path_loss_db` while preserving the same input and output dataframes.
    """

    def __init__(
        self,
        min_rssi_dbm: float = -95.0,
        min_sinr_db: float = 6.0,
        path_loss_exponent: float = 2.2,
    ) -> None:
        self.min_rssi_dbm = min_rssi_dbm
        self.min_sinr_db = min_sinr_db
        self.path_loss_exponent = path_loss_exponent

    def analyze(self, cells: pd.DataFrame, transmitters: pd.DataFrame) -> pd.DataFrame:
        if cells.empty:
            return pd.DataFrame()
        if transmitters.empty:
            raise ValueError("At least one transmitter is required")

        rows = []
        for cell in cells.itertuples(index=False):
            rx_records = []
            for tx in transmitters.itertuples(index=False):
                distance_m = _distance_m(cell, tx)
                path_loss_db = self._path_loss_db(distance_m, float(tx.frequency_mhz), cell)
                rx_power_dbm = (
                    float(tx.tx_power_dbm)
                    + float(tx.tx_gain_dbi)
                    + float(tx.rx_gain_dbi)
                    - path_loss_db
                )
                rx_records.append((tx.transmitter_id, tx.role, rx_power_dbm, tx))

            serving = max(rx_records, key=lambda item: item[2])
            signal_mw = _dbm_to_mw(serving[2])
            interference_mw = sum(
                _dbm_to_mw(power)
                for _, role, power, _ in rx_records
                if role == "jammer" or power != serving[2]
            )
            noise_dbm = _noise_dbm(float(serving[3].bandwidth_mhz), float(serving[3].noise_figure_db))
            noise_mw = _dbm_to_mw(noise_dbm)
            sinr_db = 10.0 * math.log10(signal_mw / max(interference_mw + noise_mw, 1e-15))
            interference_dbm = _mw_to_dbm(interference_mw) if interference_mw > 0 else -140.0
            stability_score = _stability_score(serving[2], sinr_db, self.min_rssi_dbm, self.min_sinr_db)
            avoid = serving[2] < self.min_rssi_dbm or sinr_db < self.min_sinr_db
            rows.append(
                {
                    "grid_id": cell.grid_id,
                    "height_layer": int(cell.height_layer),
                    "best_transmitter_id": serving[0],
                    "rssi_dbm": round(float(serving[2]), 6),
                    "sinr_db": round(float(sinr_db), 6),
                    "noise_dbm": round(float(noise_dbm), 6),
                    "interference_dbm": round(float(interference_dbm), 6),
                    "interference_level": _interference_level(interference_dbm),
                    "communication_stability": round(stability_score, 6),
                    "avoid_flag": int(avoid),
                    "recommended_action": _action(avoid, sinr_db, serving[2]),
                }
            )
        return pd.DataFrame(rows)

    def _path_loss_db(self, distance_m: float, frequency_mhz: float, cell: object) -> float:
        distance_km = max(distance_m / 1000.0, 0.001)
        fspl = 32.44 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(distance_km)
        excess = 10.0 * (self.path_loss_exponent - 2.0) * math.log10(max(distance_m, 1.0))
        terrain_loss = float(getattr(cell, "terrain_loss_db", 0.0) or 0.0)
        building_loss = float(getattr(cell, "building_loss_db", 0.0) or 0.0)
        weather_loss = min(8.0, float(getattr(cell, "weather_wind", 0.0) or 0.0) * 0.08)
        return fspl + excess + terrain_loss + building_loss + weather_loss


def _distance_m(cell: object, tx: object) -> float:
    dx = float(cell.center_x_m) - float(tx.x_m)
    dy = float(cell.center_y_m) - float(tx.y_m)
    dz = float(cell.center_z_m) - float(tx.z_m)
    return max(1.0, math.sqrt(dx * dx + dy * dy + dz * dz))


def _dbm_to_mw(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)


def _mw_to_dbm(mw: float) -> float:
    return 10.0 * math.log10(max(mw, 1e-15))


def _noise_dbm(bandwidth_mhz: float, noise_figure_db: float) -> float:
    bandwidth_hz = max(bandwidth_mhz, 1e-6) * 1_000_000.0
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def _stability_score(rssi_dbm: float, sinr_db: float, min_rssi_dbm: float, min_sinr_db: float) -> float:
    rssi_score = np.clip((rssi_dbm - min_rssi_dbm) / 35.0, 0.0, 1.0)
    sinr_score = np.clip((sinr_db - min_sinr_db) / 24.0, 0.0, 1.0)
    return float(0.45 * rssi_score + 0.55 * sinr_score)


def _interference_level(interference_dbm: float) -> int:
    if interference_dbm > -65:
        return 3
    if interference_dbm > -80:
        return 2
    if interference_dbm > -95:
        return 1
    return 0


def _action(avoid: bool, sinr_db: float, rssi_dbm: float) -> str:
    if not avoid:
        return "normal"
    if sinr_db < 0:
        return "reroute_or_change_frequency"
    if rssi_dbm < -105:
        return "add_relay_or_raise_altitude"
    return "reduce_load_or_adjust_route"

