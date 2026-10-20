"""Five-year NO2 price fan for each German price scenario in config.toml.

Writes to results/:
  forecast_NO2_weekly.csv   weekly price per inflow path and scenario
  forecast_NO2_yearly.csv   yearly fan (p10, median, p90, min, max) per scenario
  fig_forecast_fan.png      weekly fan for the first scenario
  fig_forecast_scenarios.png  yearly median and range for all scenarios

Usage:  python scripts/run_forecast.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nhm.backtest import load_config, min_fill_curve, params_from_config  # noqa: E402
from nhm.forecast import fan_table, price_fan  # noqa: E402

BLUE, GREY, ORANGE = "#1f5fa8", "#9a9a9a", "#d08a2c"


def main() -> None:
    cfg = load_config(ROOT / "config.toml")
    inflow = pd.read_csv(ROOT / "data" / "processed" / "inflow_NO2_weekly.csv")
    market = pd.read_csv(ROOT / "data" / "processed" / "market_NO2_weekly.csv")
    smax = cfg["reservoir"]["capacity_gwh"]
    r = cfg["risk"]
    curve = min_fill_curve(ROOT / "data" / "raw" / "nve_reservoir_NO2.csv",
                           tuple(r["min_fill_years"]), r["min_fill_quantile"], smax)
    p = params_from_config(cfg, curve)

    sims, tables = [], {}
    for sc in cfg["forecast"]["scenarios"]:
        sim = price_fan(cfg, inflow, market, sc["de_price"], p)
        sim.insert(0, "scenario", sc["name"])
        sims.append(sim)
        tables[sc["name"]] = fan_table(sim)
        print(f"\n{sc['name']}  (EUR/MWh, yearly mean)")
        print(tables[sc["name"]].to_string())

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    allsim = pd.concat(sims, ignore_index=True)
    allsim.to_csv(out / "forecast_NO2_weekly.csv", index=False)
    pd.concat(tables, names=["scenario", "year"]).to_csv(out / "forecast_NO2_yearly.csv")

    # Weekly fan, first scenario
    first = cfg["forecast"]["scenarios"][0]["name"]
    s = allsim[allsim["scenario"] == first]
    wk = s.groupby("week")
    x = cfg["forecast"]["start_year"] + wk["week"].first() / 52
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, wk["price"].min(), wk["price"].max(), color=BLUE, alpha=0.12, lw=0,
                    label="Range across inflow paths")
    ax.fill_between(x, wk["price"].quantile(0.1), wk["price"].quantile(0.9), color=BLUE, alpha=0.25, lw=0,
                    label="10th to 90th percentile")
    ax.plot(x, wk["price"].median(), color=BLUE, lw=1.6, label="NO2 median")
    ax.plot(x, wk["price_de"].first(), color=GREY, lw=1, label="German price (input)")
    ax.set_ylabel("EUR/MWh, weekly mean")
    fig.suptitle(f"NO2 price fan {int(x.min())} to {int(x.max())}: hydrology uncertainty only", fontsize=11, y=0.98)
    ax.set_title(f"German price: {first}. Ten historical inflow sequences, risk-averse water values.",
                 fontsize=8.5, color="#555555")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_forecast_fan.png", dpi=150)

    # Scenario comparison, yearly
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [BLUE, ORANGE, "#4c9a6a", "#8a5ca8"]
    for (name, t), c in zip(tables.items(), colors):
        ax.plot(t.index, t["no2_median"], color=c, lw=1.8, marker="o", label=f"NO2, {name}")
        ax.fill_between(t.index, t["no2_min"], t["no2_max"], color=c, alpha=0.15, lw=0)
        ax.plot(t.index, t["german"], color=c, lw=1, ls="--", label=f"German, {name}")
    ax.set_ylabel("EUR/MWh, yearly mean")
    ax.set_xticks(list(tables[first].index))
    fig.suptitle("The German price moves NO2 far more than hydrology does", fontsize=11, y=0.98)
    ax.set_title("Solid: NO2 median, shaded: range across ten inflow paths. Dashed: German input.",
                 fontsize=8.5, color="#555555")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_forecast_scenarios.png", dpi=150)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()