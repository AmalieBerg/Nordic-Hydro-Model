"""Layer 2 valuation: what is extra turbine, a pump or a battery worth?

Needs results/backtest_NO2_weekly.csv (run scripts/run_backtest.py first)
and data/processed/prices_NO2_hourly.csv (scripts/export_hourly_prices.py).

The plant's weekly water budget is its share of NO2's reservoir release
in the risk-averse layer 1 run. Share = base turbine MW / NO2's effective
reservoir capacity (max weekly release / 168 h). Variants keep the same
water, so any gain comes from timing, pumping, battery trading or less spill.

Writes:
  results/valuation_annual.csv   yearly revenue per variant
  results/valuation_summary.csv  average yearly gain vs the base plant
  results/fig_valuation.png      gain per variant, per year

Usage:  python scripts/run_valuation.py
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nhm.backtest import load_config  # noqa: E402
from nhm.hourly import Plant, annual_value, dispatch_year  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "config.toml")
    years = cfg["valuation"]["years"]

    prices = pd.read_csv(ROOT / "data" / "processed" / "prices_NO2_hourly.csv", index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert("Europe/Oslo")
    prices = prices["price_no2"]
    prices = prices[prices.index.isocalendar()["year"].isin(years).to_numpy()]

    bt = pd.read_csv(ROOT / "results" / "backtest_NO2_weekly.csv")
    bt = bt[(bt["variant"] == "risk_averse") & bt["iso_year"].isin(years)]
    no2_mw = cfg["reservoir"]["max_release_gwh"] * 1000 / 168
    share = cfg["plant"]["turbine_mw"] / no2_mw
    budgets = (bt.set_index(["iso_year", "iso_week"])["release"] * 1000 * share).rename("budget_mwh")

    base = Plant(turbine_mw=cfg["plant"]["turbine_mw"],
                 pump_efficiency=cfg["plant"]["pump_efficiency"],
                 battery_roundtrip=cfg["plant"]["battery_roundtrip"])
    plants = {"Base plant": base}
    for v in cfg["variants"]:
        plants[v["name"]] = replace(base, **{k: val for k, val in v.items() if k != "name"})

    annual = {}
    for name, plant in plants.items():
        annual[name] = annual_value(dispatch_year(prices, budgets, plant))
        print(f"{name}: done")
    annual = pd.concat(annual, names=["variant", "iso_year"])

    base_net = annual.loc["Base plant", "net_keur"]
    gain = annual["net_keur"].unstack("iso_year").sub(base_net, axis=1).drop(index="Base plant")
    summary = pd.DataFrame({
        "gain_keur_per_year": gain.mean(axis=1),
        "min_year_keur": gain.min(axis=1),
        "max_year_keur": gain.max(axis=1),
    }).round(0)

    out = ROOT / "results"
    annual.round(2).to_csv(out / "valuation_annual.csv")
    summary.to_csv(out / "valuation_summary.csv")
    print(f"\nPlant share of NO2 reservoir release: {share:.1%}")
    print(annual[["net_keur", "captured_price", "mean_price", "spill_gwh"]].round(1).to_string())
    print("\nGain over the base plant, thousand EUR per year:")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    (gain.T / 1000).plot(kind="bar", ax=ax, width=0.75, color=["#1f5fa8", "#d08a2c", "#4c9a6a"])
    ax.set_ylabel("Gain vs base plant, million EUR/year")
    ax.set_xlabel("")
    fig.suptitle("Value of flexibility for a 1,000 MW NO2 plant, same water, actual hourly prices",
                 fontsize=11, y=0.98)
    ax.set_title("Upper bound: perfect price foresight within each week, no ramping or start-up costs",
                 fontsize=8.5, color="#555555")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "fig_valuation.png", dpi=150)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()