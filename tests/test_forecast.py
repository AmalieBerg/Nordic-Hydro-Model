"""Tests for the price fan helpers."""
import numpy as np
import pandas as pd

from nhm.forecast import fan_table, inflow_paths


def test_inflow_paths_follow_history_and_wrap():
    Q = np.arange(3)[:, None] * np.ones((3, 52))  # year k has inflow k every week
    paths = inflow_paths(Q, 2)
    assert paths.shape == (3, 104)
    assert (paths[2, :52] == 2).all() and (paths[2, 52:] == 0).all()  # 2 then wraps to 0


def test_fan_table_is_ordered():
    rng = np.random.default_rng(0)
    rows = []
    for path in range(10):
        for year in (2027, 2028):
            for week in range(52):
                rows.append(dict(year=year, path=str(path), price=50 + rng.normal(0, 5), price_de=60.0))
    t = fan_table(pd.DataFrame(rows))
    assert (t["no2_min"] <= t["no2_p10"]).all()
    assert (t["no2_p10"] <= t["no2_median"]).all()
    assert (t["no2_median"] <= t["no2_p90"]).all()
    assert (t["no2_p90"] <= t["no2_max"]).all()
    assert np.allclose(t["german"], 60.0)