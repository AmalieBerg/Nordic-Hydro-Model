"""Layer 2: hourly dispatch of one plant within each week.

Layer 1 decides how much water NO2 uses each week. Layer 2 takes the
plant's share of that water and places it in the hours that pay most.
It also lets the plant pump water back up and run a battery.

For one week of H hours, with hourly price p_h (EUR/MWh):

    maximise   sum_h p_h * (g_h - u_h + d_h - c_h)

    water:     sum_h g_h - eta_pump * sum_h u_h = E      (weekly budget, MWh)
    turbine:   0 <= g_h <= P_turbine
    pump:      0 <= u_h <= P_pump
    battery:   soc_h = soc_{h-1} + eta_c * c_h - d_h / eta_d
               0 <= soc_h <= E_battery,  0 <= c_h, d_h <= P_battery
               soc at the start and end of the week = half full

g is generation, u pumping load, c and d battery charge and discharge.

The plant is a price-taker: its own dispatch does not move the price.
That is reasonable for a single plant, not for all of NO2 together,
which is why layer 1 handles the system and layer 2 the asset.

If the weekly budget is more than the turbines can produce in 168
hours, the excess is spilled and reported. That is how a larger
turbine earns money in the snowmelt.

Simplifications, stated in the README: the lower reservoir for pumping
is unlimited, there are no start-up costs or ramping limits, and the
battery is co-located but trades the same price as the plant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, diags, eye, hstack, lil_matrix


@dataclass
class Plant:
    turbine_mw: float = 1000.0
    pump_mw: float = 0.0
    pump_efficiency: float = 0.75      # MWh of water gained per MWh pumped
    battery_mw: float = 0.0
    battery_mwh: float = 0.0
    battery_roundtrip: float = 0.88

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.battery_roundtrip))


def dispatch_week(price: np.ndarray, budget_mwh: float, plant: Plant) -> dict:
    """Solve one week. Returns hourly schedule and revenue split."""
    H = len(price)
    budget = min(budget_mwh, plant.turbine_mw * H)
    spill = budget_mwh - budget
    has_pump = plant.pump_mw > 0
    has_bat = plant.battery_mw > 0 and plant.battery_mwh > 0

    # Variable order: g (H), u (H), c (H), d (H), soc (H)
    n = 5 * H
    cost = np.concatenate([-price, price, price, -price, np.zeros(H)])

    # Water balance: sum g - eta_pump * sum u = budget
    a_water = np.concatenate([np.ones(H), -plant.pump_efficiency * np.ones(H), np.zeros(3 * H)])

    # Battery state: soc_h - soc_{h-1} - eta*c_h + d_h/eta = 0, soc_{-1} = soc0
    soc0 = 0.5 * plant.battery_mwh
    I = eye(H, format="csr")
    shift = diags([np.ones(H - 1)], [-1], shape=(H, H), format="csr")
    z = csr_matrix((H, H))
    a_soc = hstack([z, z, -plant.eta * I, I / plant.eta, I - shift], format="csr")
    b_soc = np.zeros(H)
    b_soc[0] = soc0
    # End-of-week state equals the start
    a_end = lil_matrix((1, n))
    a_end[0, 5 * H - 1] = 1.0

    A_eq = bmat([[csr_matrix(a_water)], [a_soc], [a_end.tocsr()]], format="csr")
    b_eq = np.concatenate([[budget], b_soc, [soc0]])

    bounds = (
        [(0, plant.turbine_mw)] * H
        + [(0, plant.pump_mw if has_pump else 0)] * H
        + [(0, plant.battery_mw if has_bat else 0)] * H
        + [(0, plant.battery_mw if has_bat else 0)] * H
        + [(0, plant.battery_mwh)] * H
    )
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Week LP failed: {res.message}")

    x = res.x
    g, u, c, d = x[:H], x[H:2 * H], x[2 * H:3 * H], x[3 * H:4 * H]
    return dict(
        hydro_revenue=float(price @ g),
        pump_cost=float(price @ u),
        battery_margin=float(price @ (d - c)),
        generation_mwh=float(g.sum()),
        pumped_mwh=float(u.sum()),
        spill_mwh=float(spill),
        captured_price=float(price @ g / g.sum()) if g.sum() > 0 else np.nan,
        week_mean_price=float(price.mean()),
    )


def dispatch_year(prices: pd.Series, budgets: pd.Series, plant: Plant) -> pd.DataFrame:
    """Run every ISO week of hourly `prices` against the weekly `budgets`.

    prices: hourly, tz-aware index. budgets: MWh, indexed by (iso_year, iso_week).
    """
    iso = prices.index.isocalendar()
    rows = []
    for (y, w), p in prices.groupby([iso["year"], iso["week"]]):
        if (y, w) not in budgets.index or len(p) < 160:
            continue
        p = p.interpolate().bfill().ffill().to_numpy()
        out = dispatch_week(p, float(budgets.loc[(y, w)]), plant)
        rows.append(dict(iso_year=y, iso_week=w, hours=len(p), **out))
    return pd.DataFrame(rows)


def annual_value(weekly: pd.DataFrame) -> pd.DataFrame:
    """Yearly totals in thousand EUR."""
    g = weekly.groupby("iso_year")
    out = pd.DataFrame({
        "hydro_revenue_keur": g["hydro_revenue"].sum() / 1000,
        "pump_cost_keur": g["pump_cost"].sum() / 1000,
        "battery_margin_keur": g["battery_margin"].sum() / 1000,
        "spill_gwh": g["spill_mwh"].sum() / 1000,
        "captured_price": (g.apply(lambda x: (x["captured_price"] * x["generation_mwh"]).sum() / x["generation_mwh"].sum())),
        "mean_price": g["week_mean_price"].mean(),
    })
    out["net_keur"] = out["hydro_revenue_keur"] - out["pump_cost_keur"] + out["battery_margin_keur"]
    return out