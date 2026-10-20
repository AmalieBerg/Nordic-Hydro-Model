"""Grid search for the risk settings in layer 1.

Calibrates on 2020-2022 and reports 2023-2025 as out-of-sample, so the
chosen setting is not fitted to the years it is judged on.
Writes results/risk_calibration.csv.  Takes about a minute.

Usage:  python scripts/calibrate_risk.py
"""
import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from nhm.backtest import load_config, metrics, min_fill_curve, run_year  # noqa: E402

CALIBRATE, EVALUATE = [2020, 2021, 2022], [2023, 2024, 2025]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
PENALTIES = [0.0, 30.0, 100.0, 300.0]


def main() -> None:
    cfg0 = load_config(ROOT / "config.toml")
    inflow = pd.read_csv(ROOT / "data" / "processed" / "inflow_NO2_weekly.csv")
    market = pd.read_csv(ROOT / "data" / "processed" / "market_NO2_weekly.csv")
    smax = cfg0["reservoir"]["capacity_gwh"]
    r = cfg0["risk"]
    curve = min_fill_curve(ROOT / "data" / "raw" / "nve_reservoir_NO2.csv",
                           tuple(r["min_fill_years"]), r["min_fill_quantile"], smax)
    rows = []
    for w, pen in itertools.product(WEIGHTS, PENALTIES):
        cfg = copy.deepcopy(cfg0)
        cfg["risk"].update(cvar_weight=w, min_fill_penalty=pen)
        sim = pd.concat([run_year(cfg, inflow, market, y, curve) for y in CALIBRATE + EVALUATE])
        m = metrics(sim, smax)
        cal, ev = m.loc[CALIBRATE], m.loc[EVALUATE]
        rows.append(dict(cvar_weight=w, min_fill_penalty=pen,
                         cal_fill_mae_pct=cal["fill_mae_pct"].mean(), cal_price_mae=cal["mae"].mean(),
                         eval_fill_mae_pct=ev["fill_mae_pct"].mean(), eval_price_mae=ev["mae"].mean()))
        print(f"w={w:.2f} penalty={pen:>5.0f}  calibration fill error {rows[-1]['cal_fill_mae_pct']:.1f} pp")
    res = pd.DataFrame(rows).round(2).sort_values("cal_fill_mae_pct")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "risk_calibration.csv", index=False)
    print("\n", res.to_string(index=False))


if __name__ == "__main__":
    main()