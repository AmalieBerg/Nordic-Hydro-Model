"""Conditional backtest of layer 1 for 2020 to 2025, risk-neutral and risk-averse.

Writes to results/:
  backtest_NO2_weekly.csv     weekly values for both model variants
  backtest_metrics.csv        yearly error metrics for both variants
  fig_backtest_price.png      weekly NO2 price, risk-averse model vs actual
  fig_backtest_filling.png    reservoir filling: actual, risk-neutral, risk-averse

Usage:  python scripts/run_backtest.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nhm.backtest import load_config, metrics, min_fill_curve, run_year  # noqa: E402

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
AVERSE, NEUTRAL, ACTUAL, DE = "#1f5fa8", "#d08a2c", "#2b2b2b", "#b0b0b0"


def _x(df: pd.DataFrame) -> pd.Series:
    return df["iso_year"] + (df["iso_week"] - 1) / 52


def _run(cfg: dict, inflow: pd.DataFrame, market: pd.DataFrame, min_fill, label: str) -> pd.DataFrame:
    sim = pd.concat([run_year(cfg, inflow, market, y, min_fill) for y in YEARS], ignore_index=True)
    sim.insert(0, "variant", label)
    return sim


def main() -> None:
    cfg = load_config(ROOT / "config.toml")
    inflow = pd.read_csv(ROOT / "data" / "processed" / "inflow_NO2_weekly.csv")
    market = pd.read_csv(ROOT / "data" / "processed" / "market_NO2_weekly.csv")
    smax = cfg["reservoir"]["capacity_gwh"]
    r = cfg["risk"]
    curve = min_fill_curve(ROOT / "data" / "raw" / "nve_reservoir_NO2.csv",
                           tuple(r["min_fill_years"]), r["min_fill_quantile"], smax)

    neutral_cfg = copy.deepcopy(cfg)
    neutral_cfg["risk"] = {"cvar_weight": 0.0, "min_fill_penalty": 0.0}
    neutral = _run(neutral_cfg, inflow, market, None, "risk_neutral")
    averse = _run(cfg, inflow, market, curve, "risk_averse")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.concat([neutral, averse]).to_csv(out / "backtest_NO2_weekly.csv", index=False)
    m = pd.concat({"risk_neutral": metrics(neutral, smax), "risk_averse": metrics(averse, smax)})
    m.to_csv(out / "backtest_metrics.csv")
    cols = ["price_actual", "price_model", "discount_actual", "discount_model", "mae", "bias", "fill_mae_pct"]
    print(m[cols].to_string())

    x = _x(averse)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, averse["price_de"], color=DE, lw=1, label="German price (input)")
    ax.plot(x, averse["price_actual"], color=ACTUAL, lw=1.4, label="NO2 actual")
    ax.plot(x, averse["price"], color=AVERSE, lw=1.4, label="NO2 model (risk-averse)")
    ax.set_ylabel("EUR/MWh, weekly mean")
    ax.set_title("NO2 weekly price: model vs actual (conditional backtest)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_backtest_price.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, 100 * curve[averse["iso_week"].to_numpy() - 1] / smax, color=DE, lw=1, ls="--",
            label="Minimum curve (5th pct, 1995-2015)")
    ax.plot(x, 100 * averse["storage_actual"] / smax, color=ACTUAL, lw=1.4, label="Actual (NVE)")
    ax.plot(x, 100 * neutral["storage_end"] / smax, color=NEUTRAL, lw=1.2, label="Risk-neutral model")
    ax.plot(x, 100 * averse["storage_end"] / smax, color=AVERSE, lw=1.4, label="Risk-averse model")
    ax.set_ylabel("Reservoir filling, %")
    ax.set_ylim(0, 100)
    ax.set_title("NO2 reservoir filling (each year restarts from NVE's actual filling)")
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_backtest_filling.png", dpi=150)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()