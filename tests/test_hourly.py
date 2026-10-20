"""Tests for layer 2. Run with: python -m pytest -q"""
import numpy as np

from nhm.hourly import Plant, dispatch_week

H = 168
HOURS = np.arange(H)
# Cheap nights, expensive evenings
PRICE = 50 + 30 * np.sin(2 * np.pi * (HOURS - 6) / 24)


def test_water_goes_to_the_most_expensive_hours():
    plant = Plant(turbine_mw=100.0)
    budget = 100.0 * 40  # 40 full-load hours
    out = dispatch_week(PRICE, budget, plant)
    top40 = np.sort(PRICE)[-40:].sum() * 100.0
    assert np.isclose(out["hydro_revenue"], top40, rtol=1e-6)
    assert np.isclose(out["generation_mwh"], budget)
    assert out["captured_price"] > PRICE.mean()


def test_more_turbine_capacity_never_earns_less():
    budget = 100.0 * 100
    small = dispatch_week(PRICE, budget, Plant(turbine_mw=100.0))
    large = dispatch_week(PRICE, budget, Plant(turbine_mw=150.0))
    assert large["hydro_revenue"] >= small["hydro_revenue"] - 1e-6


def test_excess_water_is_spilled_when_turbines_are_full():
    out = dispatch_week(PRICE, 100.0 * H + 500.0, Plant(turbine_mw=100.0))
    assert np.isclose(out["spill_mwh"], 500.0)
    assert np.isclose(out["generation_mwh"], 100.0 * H)


def test_battery_never_loses_money_and_ends_where_it_started():
    plant = Plant(turbine_mw=100.0, battery_mw=50.0, battery_mwh=100.0)
    out = dispatch_week(PRICE, 100.0 * 40, plant)
    assert out["battery_margin"] >= -1e-6
    flat = dispatch_week(np.full(H, 50.0), 100.0 * 40, plant)
    assert abs(flat["battery_margin"]) < 1e-6  # nothing to gain on a flat price


def test_pump_adds_value_only_when_the_spread_beats_the_losses():
    pump = Plant(turbine_mw=100.0, pump_mw=50.0, pump_efficiency=0.75)
    out = dispatch_week(PRICE, 100.0 * 40, pump)
    base = dispatch_week(PRICE, 100.0 * 40, Plant(turbine_mw=100.0))
    net_pump = out["hydro_revenue"] - out["pump_cost"]
    assert net_pump >= base["hydro_revenue"] - 1e-6
    flat = dispatch_week(np.full(H, 50.0), 100.0 * 40, pump)
    assert flat["pumped_mwh"] < 1e-6  # 25% loss and no spread: never pump