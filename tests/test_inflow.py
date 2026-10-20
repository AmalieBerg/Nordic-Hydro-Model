"""Tests for the reservoir balance. Run with: python -m pytest -q"""
import numpy as np
import pandas as pd

from nhm.inflow import compute_inflow, weekly_sum_gwh


def _hourly(weeks: int, mw: float, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=weeks * 168, freq="1h", tz="Europe/Oslo")
    return pd.DataFrame({"hydro_reservoir": mw, "hydro_ror": 0.0}, index=idx)


def test_weekly_sum_is_mw_times_hours_in_gwh():
    w = weekly_sum_gwh(_hourly(2, 1000.0), ["hydro_reservoir", "hydro_ror"])
    assert np.allclose(w["hydro_reservoir"], 168.0)  # 1000 MW * 168 h = 168 GWh
    assert (w["missing_share"] == 0).all()


def test_balance_recovers_known_inflow():
    # Storage rises 50 GWh a week while 168 GWh is produced, so inflow is 218.
    prod = weekly_sum_gwh(_hourly(4, 1000.0), ["hydro_reservoir", "hydro_ror"])
    nve = pd.DataFrame(
        {"storage_gwh": [1000.0, 1050.0, 1100.0, 1150.0], "capacity_gwh": 2000.0, "filling": 0.5},
        index=prod.index,
    )
    out = compute_inflow(nve, prod)
    assert out["quality_flag"].iloc[0] == "no_previous_week"
    assert np.allclose(out["inflow_gwh"].iloc[1:], 218.0)


def test_missing_hours_are_flagged():
    h = _hourly(2, 1000.0)
    h.iloc[:20, 0] = np.nan  # 20 of 168 hours missing in week 1
    prod = weekly_sum_gwh(h, ["hydro_reservoir", "hydro_ror"])
    nve = pd.DataFrame({"storage_gwh": [1000.0, 1000.0], "capacity_gwh": 2000.0, "filling": 0.5}, index=prod.index)
    out = compute_inflow(nve, prod)
    assert out["quality_flag"].iloc[0] in {"missing_hours", "no_previous_week"}
    assert out["missing_share"].iloc[0] > 0.1


def test_missing_optional_column_does_not_flag_week():
    # Pumped storage absent for the whole period, as for NO2 before 2020.
    h = _hourly(2, 1000.0)
    h["hydro_pumped"] = np.nan
    prod = weekly_sum_gwh(h, ["hydro_reservoir", "hydro_ror", "hydro_pumped"], required=["hydro_reservoir", "hydro_ror"])
    assert (prod["missing_share"] == 0).all()
    assert (prod["hydro_pumped"] == 0).all()