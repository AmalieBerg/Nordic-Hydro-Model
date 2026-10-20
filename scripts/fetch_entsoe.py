"""Fetch NO2 and DE-LU market data from the ENTSO-E Transparency Platform.

Needs ENTSOE_API_KEY in a .env file at the repo root (see .env.example).
Pulls one calendar year at a time to stay within API limits, resamples
everything to hourly means (day-ahead moved to 15-minute MTU in 2025),
and writes parquet files to data/raw/.

Usage:  python scripts/fetch_entsoe.py            (2015 to today)
        python scripts/fetch_entsoe.py 2023 2025  (chosen years)
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
TZ = "Europe/Oslo"

# NO2 interconnectors: NordLink (DE-LU), NorNed (NL), Skagerrak (DK1),
# North Sea Link (GB), plus internal links to NO1 and NO5.
NEIGHBOURS = ["DE_LU", "NL", "DK_1", "GB", "NO_1", "NO_5"]

# ENTSO-E production types: B10 pumped storage, B11 run-of-river, B12 reservoir.
HYDRO_TYPES = {"B10": "hydro_pumped", "B11": "hydro_ror", "B12": "hydro_reservoir"}


def hourly(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return obj.tz_convert(TZ).resample("1h").mean()


def safe(label: str, fn, *args, **kwargs):
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except NoMatchingDataError:
            print(f"  no data: {label}")
            return None
        except Exception as e:  # network hiccups, rate limits
            print(f"  retry {attempt + 1} for {label}: {e}")
            time.sleep(5 * (attempt + 1))
    print(f"  FAILED: {label}")
    return None


def fetch_year(client: EntsoePandasClient, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz=TZ), pd.Timestamp.now(tz=TZ).floor("D"))
    cols = {}

    p = safe(f"price NO2 {year}", client.query_day_ahead_prices, "NO_2", start=start, end=end)
    if p is not None:
        cols["price_no2"] = hourly(p)
    p = safe(f"price DE_LU {year}", client.query_day_ahead_prices, "DE_LU", start=start, end=end)
    de = hourly(p) if p is not None else None
    if year <= 2018:
        # Before 1 Oct 2018 Germany traded in the joint DE-AT-LU bidding zone.
        old = safe(f"price DE_AT_LU {year}", client.query_day_ahead_prices, "DE_AT_LU", start=start, end=end)
        if old is not None:
            de = hourly(old) if de is None else de.combine_first(hourly(old))
    if de is not None:
        cols["price_de"] = de

    load = safe(f"load NO2 {year}", client.query_load, "NO_2", start=start, end=end)
    if load is not None:
        cols["load_no2"] = hourly(load.iloc[:, 0] if isinstance(load, pd.DataFrame) else load)

    for code, name in HYDRO_TYPES.items():
        g = safe(f"{name} {year}", client.query_generation, "NO_2", start=start, end=end, psr_type=code)
        if g is not None:
            g = g.iloc[:, 0] if isinstance(g, pd.DataFrame) else g
            cols[name] = hourly(g)

    for nb in NEIGHBOURS:
        exp = safe(f"flow NO2->{nb} {year}", client.query_crossborder_flows, "NO_2", nb, start=start, end=end)
        imp = safe(f"flow {nb}->NO2 {year}", client.query_crossborder_flows, nb, "NO_2", start=start, end=end)
        if exp is not None:
            cols[f"flow_no2_to_{nb.lower()}"] = hourly(exp)
        if imp is not None:
            cols[f"flow_{nb.lower()}_to_no2"] = hourly(imp)

    return pd.DataFrame(cols)


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.getenv("ENTSOE_API_KEY")
    if not key:
        raise SystemExit("Set ENTSOE_API_KEY in .env (copy .env.example).")
    client = EntsoePandasClient(api_key=key)

    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (2015, pd.Timestamp.now().year)
    OUT.mkdir(parents=True, exist_ok=True)

    for year in range(first, last + 1):
        print(f"Fetching {year} ...")
        df = fetch_year(client, year)
        path = OUT / f"entsoe_no2_{year}.parquet"
        df.to_parquet(path)
        print(f"  saved {path.name}: {len(df)} hours, {len(df.columns)} columns")


if __name__ == "__main__":
    main()