"""Layer 1: weekly water values by stochastic dynamic programming.

One aggregate reservoir for NO2. Each week the system chooses how much
reservoir hydro to release (R, GWh). The cables to the continent clear
the rest at the German price:

    X = R - D              net export to the continent (GWh)
    -imax <= X <= xmax     cable limits for the week
    short = max(0, D - imax - R)   unserved demand, priced at VOLL

    V_w(S) = E_q [ max_R  p_w * X - voll * short + beta * V_{w+1}(S') ]
    S'     = min(S + q - R, smax)  (anything above smax is spilled)

D is the residual demand (load net of run-of-river, other generation
and internal exchange), p_w the German price, q the inflow scenario.

The water value is lambda_w(S) = beta * dV_{w+1}/dS, in EUR/MWh.
The area price follows from which constraint binds:
  * no cable at its limit      -> price = German price
  * export cable at its limit  -> price = water value (below German)
  * import cable at its limit  -> price = water value (above German)
  * unserved demand            -> price = VOLL

Risk aversion (optional, off by default):
  * a soft minimum filling curve: each GWh below the curve costs
    min_fill_penalty, standing in for the risk of running dry later
  * a CVaR tilt: the expectation over inflow scenarios puts extra
    weight on the worst ones
The risk-neutral model drained NO2 too far in every backtest year;
the minimum curve fixed most of that (see results/).

Units: GWh times EUR/MWh gives thousand EUR, so V is in kEUR and
dV/dS in kEUR/GWh, which equals EUR/MWh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    smax: float                 # reservoir capacity, GWh
    rmax: float                 # max reservoir production per week, GWh
    voll: float = 3000.0        # value of lost load, EUR/MWh
    discount_rate: float = 0.05 # annual
    n_storage: int = 101
    n_release: int = 81
    # Risk aversion (both off by default, which gives the risk-neutral model).
    risk_weight: float = 0.0    # weight on the CVaR tail in the Bellman expectation
    cvar_alpha: float = 0.2     # tail share of inflow scenarios
    min_fill_penalty: float = 0.0           # EUR/MWh below the minimum curve
    min_fill: np.ndarray | None = field(default=None, repr=False)  # GWh by week

    def smin(self, w: int) -> float:
        if self.min_fill is None or self.min_fill_penalty <= 0:
            return 0.0
        return float(self.min_fill[w % len(self.min_fill)])

    @property
    def beta(self) -> float:
        return (1.0 + self.discount_rate) ** (-1.0 / 52.0)

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(0.0, self.smax, self.n_storage)


def _interp(grid: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(x, grid, values)


def _candidates(d: float, xmax: float, imax: float, p: Params, n: int) -> np.ndarray:
    """Release candidates: an even grid plus the exact cable-limit points."""
    base = np.linspace(0.0, p.rmax, n)
    extra = np.array([d + xmax, d - imax])
    extra = extra[(extra > 0) & (extra < p.rmax)]
    return np.unique(np.concatenate([base, extra]))


def _risk_expectation(values: np.ndarray, p: Params) -> np.ndarray:
    """Expectation over inflow scenarios, tilted towards the worst ones.

    (1 - w) * mean + w * CVaR, where CVaR is the mean of the worst
    ceil(alpha * K) outcomes. With w = 0 this is the plain mean.
    """
    mean = values.mean(axis=1)
    if p.risk_weight <= 0:
        return mean
    k = max(1, int(np.ceil(p.cvar_alpha * values.shape[1])))
    tail = np.sort(values, axis=1)[:, :k].mean(axis=1)
    return (1 - p.risk_weight) * mean + p.risk_weight * tail


def _stage(
    v_next: np.ndarray, s: np.ndarray, q: np.ndarray, price: float, d: float,
    xmax: float, imax: float, p: Params, n_release: int, smin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Best value and release for every (storage, inflow) pair.

    s: (n_s,), q: (K,)  ->  value (n_s, K), release (n_s, K)
    """
    r = _candidates(d, xmax, imax, p, n_release)               # (n_r,)
    avail = s[:, None, None] + q[None, :, None]                # (n_s, K, 1)
    rr = r[None, None, :]
    feasible = (rr <= avail) & (rr <= d + xmax + 1e-9)
    x = np.maximum(rr - d, -imax)
    short = np.maximum(0.0, d - imax - rr)
    s_next = np.minimum(avail - rr, p.smax)
    s_next = np.clip(s_next, 0.0, p.smax)
    future = _interp(p.grid, v_next, s_next)
    # Soft minimum filling: a cost per GWh below the curve. It is not money
    # anyone pays; it stands in for the risk of running dry later.
    below = np.maximum(0.0, smin - s_next)
    total = price * x - p.voll * short - p.min_fill_penalty * below + p.beta * future
    total = np.where(feasible, total, -np.inf)
    best = total.argmax(axis=2)
    value = np.take_along_axis(total, best[..., None], axis=2)[..., 0]
    # Nothing is feasible only when d + xmax < 0: even zero release would
    # overload the export cables. Then release nothing, export at the
    # limit and let the surplus (run-of-river, wind) be curtailed.
    fallback = price * xmax + p.beta * _interp(p.grid, v_next, np.minimum(s[:, None] + q[None, :], p.smax))
    none_ok = ~np.isfinite(value)
    value = np.where(none_ok, fallback, value)
    release = np.where(none_ok, 0.0, r[best])
    return value, release


def solve(
    inflow: np.ndarray, price: np.ndarray, demand: np.ndarray,
    xmax: np.ndarray, imax: np.ndarray, p: Params,
    terminal: np.ndarray | None = None, cycles: int = 1,
) -> np.ndarray:
    """Backward recursion.

    inflow: (K, W) scenarios by week. price, demand, xmax, imax: (W,).
    Returns V with shape (W + 1, n_storage); V[W] is the terminal value.
    With cycles > 1 the W-week pattern is repeated backwards and the value
    at the start of each pass becomes the terminal of the next, which
    approximates the steady-state seasonal water values.
    """
    K, W = inflow.shape
    grid = p.grid
    v_term = np.zeros(p.n_storage) if terminal is None else terminal.copy()
    for _ in range(cycles):
        V = np.empty((W + 1, p.n_storage))
        V[W] = v_term
        for w in range(W - 1, -1, -1):
            val, _ = _stage(V[w + 1], grid, inflow[:, w], price[w], demand[w], xmax[w], imax[w],
                            p, p.n_release, smin=p.smin(w))
            V[w] = _risk_expectation(val, p)
        # Remove the level so values do not drift upwards with each cycle;
        # water values depend only on the slope.
        v_term = V[0] - V[0].min()
    return V


def water_values(V: np.ndarray, p: Params) -> np.ndarray:
    """lambda[w, s] = beta * dV_{w+1}/dS, in EUR/MWh. Shape (W, n_storage)."""
    grad = np.gradient(V[1:], p.grid, axis=1)
    return p.beta * grad


def simulate(
    V: np.ndarray, inflow: np.ndarray, price: np.ndarray, demand: np.ndarray,
    xmax: np.ndarray, imax: np.ndarray, s0: float, p: Params, n_release: int = 401,
) -> pd.DataFrame:
    """Run one inflow path through the water values and record prices."""
    W = len(inflow)
    lam = water_values(V, p)
    rows = []
    s = s0
    tol = 1e-6
    for w in range(W):
        _, r_best = _stage(V[w + 1], np.array([s]), np.array([inflow[w]]), price[w], demand[w],
                           xmax[w], imax[w], p, n_release, smin=p.smin(w))
        r = float(r_best[0, 0])
        x = max(r - demand[w], -imax[w])
        short = max(0.0, demand[w] - imax[w] - r)
        avail = s + inflow[w]
        s_next = min(avail - r, p.smax)
        spill = max(0.0, avail - r - p.smax)
        wv = float(np.interp(s_next, p.grid, lam[w]))
        if short > tol:
            area_price, regime = p.voll, "shortage"
        elif x >= xmax[w] - 1e-3:
            area_price, regime = min(wv, price[w]), "export_limit"
        elif x <= -imax[w] + 1e-3:
            area_price, regime = max(wv, price[w]), "import_limit"
        else:
            area_price, regime = price[w], "coupled"
        rows.append(dict(week=w, storage_start=s, inflow=inflow[w], release=r, export=x,
                         spill=spill, short=short, storage_end=s_next, water_value=wv,
                         price_de=price[w], price=area_price, regime=regime))
        s = s_next
    return pd.DataFrame(rows)