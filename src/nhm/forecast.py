"""Five-year NO2 price fan from layer 1.

The fan shows how much NO2 prices can vary with hydrology alone, for a
given German price path. Each path runs five consecutive historical
inflow years (2016-2020, 2017-2021, ... wrapping round), so wet and dry
years follow each other as they did in history.

Inputs that are assumptions, not forecasts:
  * German price per year (config), with the weekly shape of recent years
  * NO2 residual demand: the average weekly profile of recent years
  * cables: those in operation at the start of the horizon
  * start filling: the median end-of-year filling in recent years

Water values are solved over the whole horizon at once, with a
steady-state terminal value so the last year does not empty the
reservoir just because the model ends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhm.backtest import W, _weeks, cable_limits, inflow_matrix
from nhm.sdp import Params, simulate, solve


def weekly_shape(market: pd.DataFrame, years: list[int], col: str, relative: bool) -> np.ndarray:
    """Average weekly profile over `years`. Relative profiles average to 1."""
    rows = np.array([_weeks(market, y, col) for y in years])
    if relative:
        rows = rows / rows.mean(axis=1, keepdims=True)
    return rows.mean(axis=0)


def start_fill(inflow: pd.DataFrame, years: list[int]) -> float:
    last = inflow[inflow["iso_year"].between(*years)].sort_values(["iso_year", "iso_week"])
    return float(last.groupby("iso_year")["storage_gwh"].last().median())


def inflow_paths(Q: np.ndarray, n_years: int) -> np.ndarray:
    """Path k = years k, k+1, ..., k+n-1 (wrapping). Shape (K, n_years * W)."""
    K = Q.shape[0]
    return np.array([np.concatenate([Q[(k + j) % K] for j in range(n_years)]) for k in range(K)])


def price_fan(cfg: dict, inflow: pd.DataFrame, market: pd.DataFrame,
              de_by_year: list[float], p: Params) -> pd.DataFrame:
    fc = cfg["forecast"]
    n, y0 = fc["years"], fc["start_year"]
    T = n * W
    years = cfg["scenarios"]["inflow_years"]
    Q = inflow_matrix(inflow, years)                       # (K, 52)

    shape = weekly_shape(market, fc["shape_years"], "price_de", relative=True)
    demand_w = weekly_shape(market, fc["shape_years"], "residual_demand", relative=False)
    price = np.concatenate([lvl * shape for lvl in de_by_year])
    demand = np.tile(demand_w, n)
    caps = [cable_limits(cfg, y0 + j) for j in range(n)]
    xmax = np.concatenate([c[0] for c in caps])
    imax = np.concatenate([c[1] for c in caps])

    # Steady-state terminal value from the last year's conditions
    v_ss = solve(Q, price[-W:], demand[-W:], xmax[-W:], imax[-W:], p, cycles=cfg["grid"]["cycles"])
    terminal = v_ss[0] - v_ss[0].min()

    # Horizon solve: at every week, all historical years are possible
    Q_horizon = np.tile(Q, (1, n))                          # (K, T)
    V = solve(Q_horizon, price, demand, xmax, imax, p, terminal=terminal, cycles=1)

    s0 = start_fill(inflow, fc["start_fill_years"])
    rows = []
    for k, path in enumerate(inflow_paths(Q, n)):
        sim = simulate(V, path, price, demand, xmax, imax, s0, p, n_release=cfg["grid"]["n_release_sim"])
        sim["path"] = f"{years[k]}-{years[(k + n - 1) % len(years)]}"
        sim["year"] = y0 + sim["week"] // W
        sim["iso_week"] = sim["week"] % W + 1
        rows.append(sim)
    return pd.concat(rows, ignore_index=True)


def fan_table(sim: pd.DataFrame) -> pd.DataFrame:
    """Yearly NO2 price per path, then the spread across paths."""
    yearly = sim.groupby(["year", "path"]).agg(price=("price", "mean"), price_de=("price_de", "mean"))
    g = yearly.groupby(level="year")
    return pd.DataFrame({
        "german": g["price_de"].mean(),
        "no2_p10": g["price"].quantile(0.10),
        "no2_median": g["price"].median(),
        "no2_p90": g["price"].quantile(0.90),
        "no2_min": g["price"].min(),
        "no2_max": g["price"].max(),
    }).round(1)