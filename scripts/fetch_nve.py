"""Fetch weekly reservoir statistics from NVE (Magasinstatistikk).

Source: https://biapi.nve.no/magasinstatistikk (NLOD licence, attribute NVE).
Writes the full public dataset plus two filtered files:
  data/raw/nve_reservoir_all.csv   every area NVE publishes
  data/raw/nve_reservoir_NO.csv    Norway as a whole
  data/raw/nve_reservoir_NO2.csv   price area NO2 (Sørvest)

Usage:  python scripts/fetch_nve.py
"""
from pathlib import Path

import pandas as pd
import requests

BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def get(endpoint: str) -> list[dict]:
    r = requests.get(f"{BASE}/{endpoint}", timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(get("HentOffentligData"))
    if df.empty:
        raise SystemExit("NVE returned no data. Check the endpoint in a browser.")

    # NVE field names (as published): dato_Id, omrType, omrnr, iso_aar, iso_uke,
    # fyllingsgrad, kapasitet_TWh, fylling_TWh, ...
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(OUT / "nve_reservoir_all.csv", index=False)

    print("Area types found:", sorted(df["omrType"].unique()))
    print("Columns:", list(df.columns))

    no = df[df["omrType"] == "NO"].sort_values(["iso_aar", "iso_uke"])
    no.to_csv(OUT / "nve_reservoir_NO.csv", index=False)

    # Price areas are omrType "EL"; omrnr 2 is NO2.
    no2 = df[(df["omrType"] == "EL") & (df["omrnr"] == 2)].sort_values(["iso_aar", "iso_uke"])
    no2.to_csv(OUT / "nve_reservoir_NO2.csv", index=False)

    for name, d in [("NO", no), ("NO2", no2)]:
        if d.empty:
            print(f"Warning: no rows for {name}. Inspect nve_reservoir_all.csv.")
        else:
            print(f"{name}: {len(d)} weeks, {d['iso_aar'].min()}-{d['iso_aar'].max()}")


if __name__ == "__main__":
    main()
