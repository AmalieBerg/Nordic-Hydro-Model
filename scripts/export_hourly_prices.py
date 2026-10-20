"""Export hourly NO2 and German prices to one CSV (for sharing and for layer 2).

Usage:  python scripts/export_hourly_prices.py
Writes data/processed/prices_NO2_hourly.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhm.inflow import load_entsoe  # noqa: E402


def main() -> None:
    raw = ROOT / "data" / "raw"
    files = sorted(raw.glob("entsoe_no2_*.parquet")) or sorted(raw.glob("entsoe_no2_*.csv"))
    h = load_entsoe(files)[["price_no2", "price_de"]]
    out = ROOT / "data" / "processed" / "prices_NO2_hourly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(out, index_label="time", float_format="%.2f")
    print(f"Saved {len(h)} hours to {out}")
    print(h.isna().groupby(h.index.year).mean().round(3).to_string())


if __name__ == "__main__":
    main()