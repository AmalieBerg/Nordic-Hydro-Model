"""Conditional backtest of layer 1.

Question: given what actually happened outside the model (German prices,
NO2 demand, cable capacity, inflow), does the water value logic
reproduce NO2 prices and the reservoir path?

This separates model error from input-forecast error. The model does
not see the test year's inflow when it computes water values: that
year is left out of the scenario set, so the plan is made under the
same inflow uncertainty a real planner faces. German prices are
treated as known, which is stated plainly in the results.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from nhm.sdp import Params, simulate, solve

W = 52


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def min_fill_curve(nve_path: str | Path, years: tuple[int, int], quantile: float, smax: float) -> np.ndarray:
    """Seasonal minimum filling (GWh by week) from NVE history.

    Uses only years before the inflow scenarios start, so the curve is
    not fitted to the period the backtest evaluates.
    """
    n = pd.read_csv(nve_path)
    n = n[(n["iso_aar"] >= years[0]) & (n["iso_aar"] <= years[1]) & (n["iso_uke"] <= W)]
    q = n.groupby("iso_uke")["fyllingsgrad"].quantile(quantile).reindex(range(1, W + 1))
    return q.interpolate().to_numpy() * smax


def params_from_config(cfg: dict, min_fill: np.ndarray | None = None) -> Params:
    risk = cfg.get("risk", {})
    return Params(
        smax=cfg["reservoir"]["capacity_gwh"],
        rmax=cfg["reservoir"]["max_release_gwh"],
        voll=cfg["market"]["voll_eur_mwh"],
        discount_rate=cfg["market"]["discount_rate"],
        n_storage=cfg["grid"]["n_storage"],
        n_release=cfg["grid"]["n_release"],
        risk_weight=risk.get("cvar_weight", 0.0),
        cvar_alpha=risk.get("cvar_alpha", 0.2),
        min_fill_penalty=risk.get("min_fill_penalty", 0.0),
        min_fill=min_fill,
    )


def cable_limits(cfg: dict, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Weekly export and import limits in GWh for each ISO week of a year."""
    avail = cfg["market"]["cable_availability"]
    xmax, imax = np.zeros(W), np.zeros(W)
    for w in range(W):
        monday = pd.Timestamp.fromisocalendar(year, w + 1, 1)
        for c in cfg["cables"]:
            if pd.Timestamp(c["start"]) <= monday:
                xmax[w] += c["export_mw"]
                imax[w] += c["import_mw"]
    to_gwh = 168 / 1000 * avail
    return xmax * to_gwh, imax * to_gwh


def _weeks(df: pd.DataFrame, year: int, col: str) -> np.ndarray:
    s = df[(df["iso_year"] == year) & (df["iso_week"] <= W)].set_index("iso_week")[col]
    return s.reindex(range(1, W + 1)).interpolate().bfill().ffill().to_numpy()


def inflow_matrix(inflow: pd.DataFrame, years: list[int]) -> np.ndarray:
    return np.array([_weeks(inflow, y, "inflow_gwh") for y in years])


def start_storage(inflow: pd.DataFrame, year: int) -> float:
    prev = inflow[inflow["iso_year"] == year - 1].sort_values("iso_week")
    return float(prev["storage_gwh"].iloc[-1])


def run_year(cfg: dict, inflow: pd.DataFrame, market: pd.DataFrame, year: int,
             min_fill: np.ndarray | None = None) -> pd.DataFrame:
    p = params_from_config(cfg, min_fill)
    train = [y for y in cfg["scenarios"]["inflow_years"] if y != year]
    Q = inflow_matrix(inflow, train)
    price = _weeks(market, year, "price_de")
    demand = _weeks(market, year, "residual_demand")
    xmax, imax = cable_limits(cfg, year)

    V = solve(Q, price, demand, xmax, imax, p, cycles=cfg["grid"]["cycles"])
    sim = simulate(V, _weeks(inflow, year, "inflow_gwh"), price, demand, xmax, imax,
                   start_storage(inflow, year), p, n_release=cfg["grid"]["n_release_sim"])
    sim.insert(0, "iso_year", year)
    sim.insert(1, "iso_week", sim.pop("week") + 1)
    sim["price_actual"] = _weeks(market, year, "price_no2")
    sim["storage_actual"] = _weeks(inflow, year, "storage_gwh")
    sim["export_actual"] = _weeks(market, year, "net_export_continent")
    sim["xmax"] = xmax
    sim["imax"] = imax
    return sim


def metrics(sim: pd.DataFrame, smax: float) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        err = g["price"] - g["price_actual"]
        return pd.Series({
            "price_actual": g["price_actual"].mean(),
            "price_model": g["price"].mean(),
            "price_de": g["price_de"].mean(),
            "discount_actual": (g["price_de"] - g["price_actual"]).mean(),
            "discount_model": (g["price_de"] - g["price"]).mean(),
            "mae": err.abs().mean(),
            "bias": err.mean(),
            "corr": g["price"].corr(g["price_actual"]),
            "fill_mae_pct": 100 * (g["storage_end"] - g["storage_actual"]).abs().mean() / smax,
            "weeks_export_limit": (g["regime"] == "export_limit").sum(),
        })
    return sim.groupby("iso_year").apply(one).round(2)