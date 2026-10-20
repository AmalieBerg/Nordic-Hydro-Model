"""Build the weekly NO2 inflow series and save it to data/processed/.

Usage:  python scripts/build_inflow.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhm.inflow import missing_report, run  # noqa: E402


def main() -> None:
    raw = ROOT / "data" / "raw"
    files = sorted(raw.glob("entsoe_no2_*.parquet")) or sorted(raw.glob("entsoe_no2_*.csv"))
    print("Share of missing hours per year and column:")
    print(missing_report(files).to_string(), "\n")
    df = run(raw / "nve_reservoir_NO2.csv", files)

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "inflow_NO2_weekly.csv")

    ok = df[df["quality_flag"] == "ok"]
    yearly = ok.groupby(level="iso_year")["inflow_gwh"].agg(["sum", "count"])
    yearly["sum_TWh"] = (yearly["sum"] / 1000).round(1)
    print(f"Weeks: {len(df)}, usable: {len(ok)}")
    print(df["quality_flag"].value_counts().to_string())
    print("\nInflow per year (only usable weeks, TWh):")
    print(yearly[["sum_TWh", "count"]].to_string())
    neg = ok[ok["inflow_gwh"] < 0]
    print(f"\nWeeks with negative inflow: {len(neg)} (a sign of data or timing issues)")


if __name__ == "__main__":
    main()