"""Weekly energy inflow for a price area, backed out from a reservoir balance.

For each ISO week t:

    inflow_t = (storage_t - storage_{t-1}) + reservoir_production_t
               + run_of_river_production_t - pumping_t

All terms in GWh. Storage comes from NVE (end-of-week filling, TWh).
Production comes from ENTSO-E (hourly MW, so the weekly sum is MWh).

What this ignores, and why it matters:
  * Spill: water released without production is invisible here, so
    inflow is underestimated in wet weeks with full reservoirs.
  * Coverage: NVE measures the ~490 largest reservoirs, while ENTSO-E
    reports all production. Stations whose reservoirs are not in the
    statistics add production without a matching storage change.
  * Capacity: NVE applies today's reservoir capacity to the whole
    history, so early years are on today's capacity basis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Oslo"


def load_nve(path: str | Path) -> pd.DataFrame:
    """End-of-week reservoir filling, indexed by (iso_year, iso_week)."""
    df = pd.read_csv(path)
    df = df.rename(columns={"iso_aar": "iso_year", "iso_uke": "iso_week"})
    df["storage_gwh"] = df["fylling_TWh"] * 1000.0
    df["capacity_gwh"] = df["kapasitet_TWh"] * 1000.0
    df["filling"] = df["fyllingsgrad"]
    df = df.sort_values(["iso_year", "iso_week"]).set_index(["iso_year", "iso_week"])
    return df[["storage_gwh", "capacity_gwh", "filling"]]


def load_entsoe(paths: list[str | Path]) -> pd.DataFrame:
    """Concatenate hourly ENTSO-E files (CSV or parquet) into one frame."""
    frames = []
    for p in sorted(paths):
        p = Path(p)
        if p.suffix == ".parquet":
            f = pd.read_parquet(p)
        else:
            f = pd.read_csv(p, index_col=0)
            f.index = pd.to_datetime(f.index, utc=True)
        f.index = pd.DatetimeIndex(f.index).tz_convert(TZ)
        frames.append(f)
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def weekly_sum_gwh(
    hourly: pd.DataFrame,
    columns: list[str],
    required: list[str] | None = None,
) -> pd.DataFrame:
    """Sum hourly MW columns into GWh per ISO week (Monday to Sunday, Oslo time).

    Only the `required` columns count towards `missing_share`. Optional
    columns (such as pumped storage, which ENTSO-E reports for NO2 only
    from 2020) are summed when present and treated as zero when absent.
    """
    required = required if required is not None else columns
    iso = hourly.index.isocalendar()
    keys = [iso["year"].rename("iso_year"), iso["week"].rename("iso_week")]
    present = [c for c in columns if c in hourly.columns]
    req = [c for c in required if c in hourly.columns]
    out = hourly[present].groupby(keys).sum(min_count=1) / 1000.0
    optional = [c for c in present if c not in req]
    out[optional] = out[optional].fillna(0.0)
    n_hours = hourly.groupby(keys).size()
    n_missing = hourly[req].isna().groupby(keys).sum().max(axis=1)
    out["hours"] = n_hours
    out["missing_share"] = n_missing / n_hours
    out.index = out.index.set_names(["iso_year", "iso_week"])
    return out


def compute_inflow(nve: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """Join storage and production, and back out weekly inflow in GWh."""
    df = nve.join(prod, how="inner")
    df["d_storage_gwh"] = df["storage_gwh"].diff()
    ror = df.get("hydro_ror", 0.0)
    pump = df.get("hydro_pumped", 0.0)
    df["inflow_gwh"] = df["d_storage_gwh"] + df["hydro_reservoir"] + ror
    # ENTSO-E B10 is pumped-storage generation; pumping load is not in this
    # series, so pumping is left out and noted as a known limitation.
    df["quality_flag"] = np.where(df["missing_share"] > 0.02, "missing_hours", "ok")
    df.loc[df["d_storage_gwh"].isna(), "quality_flag"] = "no_previous_week"
    df.loc[df["hours"] < 160, "quality_flag"] = "partial_week"
    return df


def run(nve_path: str | Path, entsoe_paths: list[str | Path]) -> pd.DataFrame:
    nve = load_nve(nve_path)
    hourly = load_entsoe(entsoe_paths)
    prod = weekly_sum_gwh(
        hourly,
        ["hydro_reservoir", "hydro_ror", "hydro_pumped"],
        required=["hydro_reservoir", "hydro_ror"],
    )
    return compute_inflow(nve, prod)


def missing_report(entsoe_paths: list[str | Path]) -> pd.DataFrame:
    """Share of missing hours per year and column, to see where gaps come from."""
    hourly = load_entsoe(entsoe_paths)
    cols = [c for c in ["hydro_reservoir", "hydro_ror", "hydro_pumped", "price_no2", "load_no2"] if c in hourly.columns]
    return hourly[cols].isna().groupby(hourly.index.year).mean().round(3)