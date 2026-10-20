"""Aggregate hourly ENTSO-E data to the weekly market inputs layer 1 needs.

Output, per ISO week (GWh unless noted):
  price_no2, price_de      mean day-ahead price (EUR/MWh)
  load                     NO2 consumption
  hydro_reservoir, hydro_ror, hydro_pumped
  net_export_<nb>          exports minus imports on each link
  net_export_continent     DE-LU + NL + DK1 + GB
  net_export_internal      NO1 + NO5
  other_gen                everything the balance needs besides hydro
                           (wind, thermal, small producers, losses)
  residual_demand          what reservoir hydro and the continent cables
                           must cover:  load - ror - pumped - other_gen
                                        + net_export_internal
"""
from __future__ import annotations

import pandas as pd

CONTINENT = ["de_lu", "nl", "dk_1", "gb"]
INTERNAL = ["no_1", "no_5"]


def _net(hourly: pd.DataFrame, nb: str) -> pd.Series:
    exp = hourly.get(f"flow_no2_to_{nb}", pd.Series(0.0, index=hourly.index)).fillna(0.0)
    imp = hourly.get(f"flow_{nb}_to_no2", pd.Series(0.0, index=hourly.index)).fillna(0.0)
    return exp - imp


def weekly_market(hourly: pd.DataFrame) -> pd.DataFrame:
    iso = hourly.index.isocalendar()
    keys = [iso["year"].rename("iso_year"), iso["week"].rename("iso_week")]

    h = pd.DataFrame(index=hourly.index)
    for c in ["price_no2", "price_de"]:
        h[c] = hourly.get(c)
    for c in ["load_no2", "hydro_reservoir", "hydro_ror", "hydro_pumped"]:
        h[c] = hourly.get(c, pd.Series(0.0, index=hourly.index))
    for nb in CONTINENT + INTERNAL:
        h[f"net_export_{nb}"] = _net(hourly, nb)

    g = h.groupby(keys)
    out = pd.DataFrame({
        "price_no2": g["price_no2"].mean(),
        "price_de": g["price_de"].mean(),
        "hours": g.size(),
    })
    energy = [c for c in h.columns if c not in ("price_no2", "price_de")]
    sums = g[energy].sum(min_count=1) / 1000.0
    sums = sums.rename(columns={"load_no2": "load"})
    out = out.join(sums)
    for c in ["hydro_pumped"]:
        out[c] = out[c].fillna(0.0)

    out["net_export_continent"] = out[[f"net_export_{nb}" for nb in CONTINENT]].sum(axis=1)
    out["net_export_internal"] = out[[f"net_export_{nb}" for nb in INTERNAL]].sum(axis=1)
    net_total = out["net_export_continent"] + out["net_export_internal"]
    hydro = out["hydro_reservoir"] + out["hydro_ror"] + out["hydro_pumped"]
    out["other_gen"] = out["load"] + net_total - hydro
    out["residual_demand"] = (
        out["load"] - out["hydro_ror"] - out["hydro_pumped"] - out["other_gen"] + out["net_export_internal"]
    )
    # By construction: hydro_reservoir = residual_demand + net_export_continent.
    out.index = out.index.set_names(["iso_year", "iso_week"])
    return out