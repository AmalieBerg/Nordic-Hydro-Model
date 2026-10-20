"""Tests for the weekly market aggregation."""
import numpy as np
import pandas as pd

from nhm.weekly import weekly_market


def test_balance_identity_holds():
    idx = pd.date_range("2024-01-01", periods=2 * 168, freq="1h", tz="Europe/Oslo")
    rng = np.random.default_rng(1)
    h = pd.DataFrame({
        "price_no2": 50.0, "price_de": 70.0,
        "load_no2": 4000 + rng.normal(0, 100, len(idx)),
        "hydro_reservoir": 5000 + rng.normal(0, 200, len(idx)),
        "hydro_ror": 300.0,
        "flow_no2_to_de_lu": 1200.0, "flow_de_lu_to_no2": 0.0,
        "flow_no2_to_no_1": 200.0, "flow_no_5_to_no2": 400.0,
    }, index=idx)  # pumped storage and GB missing, as in early years
    w = weekly_market(h)
    assert np.allclose(w["hydro_reservoir"], w["residual_demand"] + w["net_export_continent"])
    assert np.allclose(w["net_export_continent"], 1200 * 168 / 1000)
    assert np.allclose(w["net_export_internal"], (200 - 400) * 168 / 1000)
    assert (w["hydro_pumped"] == 0).all()