"""Build weekly market inputs for NO2 and save them to data/processed/.

Usage:  python scripts/build_weekly.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhm.inflow import load_entsoe  # noqa: E402
from nhm.weekly import weekly_market  # noqa: E402


def main() -> None:
    raw = ROOT / "data" / "raw"
    files = sorted(raw.glob("entsoe_no2_*.parquet")) or sorted(raw.glob("entsoe_no2_*.csv"))
    w = weekly_market(load_entsoe(files))
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    w.to_csv(out / "market_NO2_weekly.csv")

    yearly = w.groupby(level="iso_year").agg(
        price_no2=("price_no2", "mean"),
        price_de=("price_de", "mean"),
        load_TWh=("load", lambda s: s.sum() / 1000),
        export_continent_TWh=("net_export_continent", lambda s: s.sum() / 1000),
        export_internal_TWh=("net_export_internal", lambda s: s.sum() / 1000),
        other_gen_TWh=("other_gen", lambda s: s.sum() / 1000),
    ).round(1)
    print(yearly.to_string())


if __name__ == "__main__":
    main()