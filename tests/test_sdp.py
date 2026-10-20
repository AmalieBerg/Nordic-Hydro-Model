"""Behavioural tests for layer 1. Run with: python -m pytest -q"""
import numpy as np

from nhm.sdp import Params, simulate, solve, water_values

W = 52
WK = np.arange(W)


def _setup(cap: float = 700.0, price_level: float = 80.0):
    rng = np.random.default_rng(0)
    seasonal = 400 + 1500 * np.exp(-0.5 * ((WK - 22) / 4.0) ** 2)
    Q = np.clip(seasonal[None, :] * rng.lognormal(0, 0.25, size=(8, W)), 0, None)
    price = price_level + 20 * np.cos(2 * np.pi * WK / 52)
    demand = np.full(W, 850.0)
    xmax = np.full(W, cap)
    imax = np.full(W, cap)
    p = Params(smax=34000.0, rmax=1450.0, n_storage=61, n_release=61)
    return Q, price, demand, xmax, imax, p


def test_water_value_falls_as_the_reservoir_fills():
    Q, price, d, xmax, imax, p = _setup()
    lam = water_values(solve(Q, price, d, xmax, imax, p, cycles=3), p)
    assert np.mean(np.diff(lam, axis=1) <= 1e-6) > 0.98


def test_mass_balance_holds_every_week():
    Q, price, d, xmax, imax, p = _setup()
    V = solve(Q, price, d, xmax, imax, p, cycles=3)
    sim = simulate(V, Q[0], price, d, xmax, imax, s0=20000.0, p=p, n_release=121)
    lhs = sim["storage_end"]
    rhs = sim["storage_start"] + sim["inflow"] - sim["release"] - sim["spill"]
    assert np.allclose(lhs, rhs, atol=1e-6)
    assert (sim["storage_end"] >= -1e-9).all() and (sim["storage_end"] <= p.smax + 1e-9).all()


def test_price_equals_german_price_when_no_cable_binds():
    Q, price, d, xmax, imax, p = _setup(cap=1e6)  # cables never bind
    V = solve(Q, price, d, xmax, imax, p, cycles=3)
    sim = simulate(V, Q[0], price, d, xmax, imax, s0=20000.0, p=p, n_release=121)
    assert (sim["regime"] == "coupled").all()
    assert np.allclose(sim["price"], sim["price_de"])


def test_isolated_wet_system_prices_below_german():
    # Tiny cables and a full reservoir: water must be used at home,
    # so the area price drops to the water value, below the German price.
    Q, price, d, xmax, imax, p = _setup(cap=50.0)
    V = solve(Q * 1.5, price, d, xmax, imax, p, cycles=3)
    sim = simulate(V, Q[0] * 1.5, price, d, xmax, imax, s0=30000.0, p=p, n_release=121)
    assert (sim["price"] <= sim["price_de"] + 1e-9).all()
    assert (sim["regime"] == "export_limit").any()


def test_zero_risk_weight_is_the_plain_mean():
    from nhm.sdp import _risk_expectation
    vals = np.array([[1.0, 2.0, 3.0, 10.0]])
    p = Params(smax=1.0, rmax=1.0, risk_weight=0.0)
    assert np.isclose(_risk_expectation(vals, p)[0], 4.0)
    p = Params(smax=1.0, rmax=1.0, risk_weight=1.0, cvar_alpha=0.5)
    assert np.isclose(_risk_expectation(vals, p)[0], 1.5)  # mean of the worst two


def test_minimum_curve_keeps_more_water_in_a_dry_year():
    Q, price, d, xmax, imax, p = _setup()
    dry = Q[0] * 0.6
    V0 = solve(Q, price, d, xmax, imax, p, cycles=3)
    neutral = simulate(V0, dry, price, d, xmax, imax, s0=15000.0, p=p, n_release=121)
    p_risk = Params(smax=p.smax, rmax=p.rmax, n_storage=p.n_storage, n_release=p.n_release,
                    min_fill_penalty=100.0, min_fill=np.full(W, 12000.0))
    V1 = solve(Q, price, d, xmax, imax, p_risk, cycles=3)
    averse = simulate(V1, dry, price, d, xmax, imax, s0=15000.0, p=p_risk, n_release=121)
    assert averse["storage_end"].min() >= neutral["storage_end"].min()
    assert averse["storage_end"].mean() > neutral["storage_end"].mean()