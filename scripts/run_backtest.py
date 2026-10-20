"""Conditional backtest of layer 1 for 2020 to 2025.

Writes:
  results/backtest_NO2_weekly.csv   weekly model and actual values
  results/backtest_metrics.csv      yearly error metrics
  results/fig_backtest_price.png    weekly NO2 price, model vs actual
  results/fig_backtest_filling.png  reservoir filling, model vs actual

Usage:  python scripts/run_backtest.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nhm.backtest import load_config, metrics, run_year  # noqa: E402

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MODEL, ACTUAL, DE = "#1f5fa8", "#2b2b2b", "#b0b0b0"


def _x(df: pd.DataFrame) -> pd.Series:
    return df["iso_year"] + (df["iso_week"] - 1) / 52


def main() -> None:
    cfg = load_config(ROOT / "config.toml")
    inflow = pd.read_csv(ROOT / "data" / "processed" / "inflow_NO2_weekly.csv")
    market = pd.read_csv(ROOT / "data" / "processed" / "market_NO2_weekly.csv")

    sim = pd.concat([run_year(cfg, inflow, market, y) for y in YEARS], ignore_index=True)
    m = metrics(sim, cfg["reservoir"]["capacity_gwh"])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    sim.to_csv(out / "backtest_NO2_weekly.csv", index=False)
    m.to_csv(out / "backtest_metrics.csv")
    print(m.to_string())

    x = _x(sim)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, sim["price_de"], color=DE, lw=1, label="German price (input)")
    ax.plot(x, sim["price_actual"], color=ACTUAL, lw=1.4, label="NO2 actual")
    ax.plot(x, sim["price"], color=MODEL, lw=1.4, label="NO2 model")
    ax.set_ylabel("EUR/MWh, weekly mean")
    ax.set_title("NO2 weekly price: model vs actual (conditional backtest)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_backtest_price.png", dpi=150)

    smax = cfg["reservoir"]["capacity_gwh"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, 100 * sim["storage_actual"] / smax, color=ACTUAL, lw=1.4, label="Actual (NVE)")
    ax.plot(x, 100 * sim["storage_end"] / smax, color=MODEL, lw=1.4, label="Model")
    ax.set_ylabel("Reservoir filling, %")
    ax.set_ylim(0, 100)
    ax.set_title("NO2 reservoir filling: model vs actual (each year restarts from NVE's actual filling)")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_backtest_filling.png", dpi=150)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()