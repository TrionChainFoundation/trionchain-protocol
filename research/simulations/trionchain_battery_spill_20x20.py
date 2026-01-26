#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrionChain 20x20 + LP market + line limits + line shock
+ Battery that charges ONLY from real spill (curtailment)

Outputs:
- Standard plots (mismatch, prices, congestion, etc.)
- Battery plots (SoC, spill_before vs spill_after)
- Saves heatmaps: prices_last.png, prices_mean.png, prices_std.png
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# -----------------------------
# Heatmap saver
# -----------------------------
def save_price_maps(prices_by_step: List[np.ndarray], W: int, H: int, out_prefix: str = "prices") -> None:
    P = np.stack(prices_by_step, axis=0)      # (T, N)
    P2 = P.reshape(P.shape[0], H, W)          # (T, H, W)

    last_map = P2[-1]
    mean_map = P2.mean(axis=0)
    std_map = P2.std(axis=0)

    def plot_map(M: np.ndarray, title: str, fname: str) -> None:
        plt.figure(figsize=(7, 6))
        plt.imshow(M, origin="lower", aspect="equal")
        plt.colorbar(label="Price")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(fname, dpi=160)
        plt.close()

    plot_map(last_map, "Price map (last step)", f"{out_prefix}_last.png")
    plot_map(mean_map, "Price map (mean over time)", f"{out_prefix}_mean.png")
    plot_map(std_map,  "Price map (std over time)",  f"{out_prefix}_std.png")


# -----------------------------
# Config
# -----------------------------
@dataclass
class SimConfig:
    W: int = 20
    H: int = 20
    T: int = 60
    seed: int = 7

    # Network
    base_edge_cap: float = 40.0

    # Shock (vertical corridor)
    shock_t0: int = 18
    shock_t1: int = 40
    shock_col: int = 10
    shock_factor: float = 0.25

    # Node load/gen shaping
    base_load_mw: float = 2.2
    base_gen_cap_mw: float = 6.0

    # LP costs
    gen_cost: float = 10.0
    unmet_penalty: float = 10_000.0
    spill_penalty: float = 1.0

    # Price heuristic
    scarcity_price_k: float = 60.0
    congestion_price_k: float = 25.0

    # Battery (GLOBAL, charges only from spill)
    dt_hours: float = 1.0
    batt_E_max_mwh: float = 500.0
    batt_P_max_mw: float = 80.0
    batt_eta_c: float = 0.95


# -----------------------------
# Grid helpers
# -----------------------------
def node_id(x: int, y: int, W: int) -> int:
    return y * W + x

def id_xy(i: int, W: int) -> Tuple[int, int]:
    return (i % W, i // W)

def build_grid_edges(W: int, H: int) -> List[Tuple[int, int]]:
    edges = []
    for y in range(H):
        for x in range(W):
            u = node_id(x, y, W)
            if x + 1 < W:
                v = node_id(x + 1, y, W)
                edges.append((u, v))
            if y + 1 < H:
                v = node_id(x, y + 1, W)
                edges.append((u, v))
    return edges


# -----------------------------
# Shocked capacities
# -----------------------------
def apply_line_limit_shock(
    W: int,
    edges: List[Tuple[int, int]],
    base_caps: np.ndarray,
    t: int,
    cfg: SimConfig,
) -> np.ndarray:
    caps = base_caps.copy()
    if cfg.shock_t0 <= t <= cfg.shock_t1:
        c = cfg.shock_col
        for e, (u, v) in enumerate(edges):
            ux, uy = id_xy(u, W)
            vx, vy = id_xy(v, W)
            if uy == vy and ((ux == c - 1 and vx == c) or (ux == c and vx == c - 1)):
                caps[e] = base_caps[e] * cfg.shock_factor
    return caps


# -----------------------------
# LP: flows + gen + unmet + spill
# -----------------------------
def solve_flow_lp(
    W: int,
    H: int,
    edges: List[Tuple[int, int]],
    edge_caps: np.ndarray,
    load: np.ndarray,
    gen_cap: np.ndarray,
    cfg: SimConfig,
) -> Dict[str, np.ndarray]:
    N = W * H
    E = len(edges)

    idx_g0 = 0
    idx_u0 = idx_g0 + N
    idx_s0 = idx_u0 + N
    idx_f0 = idx_s0 + N
    nvars = idx_f0 + E

    c = np.zeros(nvars, dtype=float)
    c[idx_g0:idx_g0 + N] = cfg.gen_cost
    c[idx_u0:idx_u0 + N] = cfg.unmet_penalty
    c[idx_s0:idx_s0 + N] = cfg.spill_penalty

    A_eq = np.zeros((N, nvars), dtype=float)
    b_eq = load.astype(float).copy()

    # g + u - s
    for i in range(N):
        A_eq[i, idx_g0 + i] = 1.0
        A_eq[i, idx_u0 + i] = 1.0
        A_eq[i, idx_s0 + i] = -1.0

    # flow incidence: f>0 means u->v
    for e, (u, v) in enumerate(edges):
        j = idx_f0 + e
        A_eq[u, j] += -1.0
        A_eq[v, j] += +1.0

    bounds: List[Tuple[float, float]] = []
    bounds.extend([(0.0, float(gen_cap[i])) for i in range(N)])          # g
    bounds.extend([(0.0, float(load[i])) for i in range(N)])             # u
    bounds.extend([(0.0, float(gen_cap[i])) for i in range(N)])          # s
    bounds.extend([(-float(edge_caps[e]), float(edge_caps[e])) for e in range(E)])  # f

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    g = x[idx_g0:idx_g0 + N]
    u = x[idx_u0:idx_u0 + N]
    s = x[idx_s0:idx_s0 + N]
    f = x[idx_f0:idx_f0 + E]

    util_edges = np.abs(f) / (edge_caps + 1e-12)
    avg_util = float(np.mean(util_edges))

    total_load = float(load.sum())
    unmet_ratio = float(u.sum()) / (total_load + 1e-12)

    node_util = np.zeros(N, dtype=float)
    deg = np.zeros(N, dtype=float)
    for e, (a, b) in enumerate(edges):
        ue = util_edges[e]
        node_util[a] += ue
        node_util[b] += ue
        deg[a] += 1.0
        deg[b] += 1.0
    node_util = node_util / (deg + 1e-12)

    prices = cfg.gen_cost + cfg.scarcity_price_k * unmet_ratio + cfg.congestion_price_k * node_util
    avg_price = float(cfg.gen_cost + cfg.scarcity_price_k * unmet_ratio + cfg.congestion_price_k * avg_util)

    return {
        "gen": g,
        "unmet": u,
        "spill": s,
        "flow": f,
        "avg_util": np.array([avg_util], dtype=float),
        "unmet_ratio": np.array([unmet_ratio], dtype=float),
        "prices": prices,
        "avg_price": np.array([avg_price], dtype=float),
    }


# -----------------------------
# Metrics
# -----------------------------
def boundary_mismatch(W: int, edges: List[Tuple[int, int]], flows: np.ndarray, shock_col: int) -> float:
    net = 0.0
    c = shock_col
    for e, (u, v) in enumerate(edges):
        ux, uy = id_xy(u, W)
        vx, vy = id_xy(v, W)
        if uy == vy and ((ux == c - 1 and vx == c) or (ux == c and vx == c - 1)):
            if ux == c - 1 and vx == c:
                net += flows[e]
            else:
                net -= flows[e]
    return float(abs(net))


# -----------------------------
# Battery: charge ONLY from spill
# -----------------------------
def battery_charge_from_spill(SoC_mwh: float, spill_mw: float, cfg: SimConfig) -> Tuple[float, float, float]:
    """
    Returns:
      new_SoC_mwh
      charge_mw (power drawn from spill)
      spill_remaining_mw
    """
    dt = cfg.dt_hours

    # available spill energy this step (MWh)
    spill_mwh = spill_mw * dt

    # max charge energy limited by power and remaining capacity
    max_charge_mwh_by_power = cfg.batt_P_max_mw * dt
    max_charge_mwh_by_space = max(0.0, cfg.batt_E_max_mwh - SoC_mwh)

    # Energy that can be stored (after efficiency):
    # stored = (charge_from_spill) * eta_c
    # so charge_from_spill = stored / eta_c
    max_stored_mwh = min(max_charge_mwh_by_power * cfg.batt_eta_c, max_charge_mwh_by_space)
    charge_from_spill_mwh = min(spill_mwh, max_stored_mwh / cfg.batt_eta_c)

    stored_mwh = charge_from_spill_mwh * cfg.batt_eta_c
    new_soc = SoC_mwh + stored_mwh

    spill_remaining_mwh = max(0.0, spill_mwh - charge_from_spill_mwh)
    # convert back to MW (per dt)
    charge_mw = charge_from_spill_mwh / dt
    spill_remaining_mw = spill_remaining_mwh / dt

    return new_soc, charge_mw, spill_remaining_mw


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = SimConfig()
    rng = np.random.default_rng(cfg.seed)

    W, H, T = cfg.W, cfg.H, cfg.T
    N = W * H

    edges = build_grid_edges(W, H)
    E = len(edges)
    base_caps = np.full(E, cfg.base_edge_cap, dtype=float)

    base_load_field = cfg.base_load_mw * (0.9 + 0.2 * rng.random(N))
    gen_cap_field = cfg.base_gen_cap_mw * (0.9 + 0.2 * rng.random(N))

    # Make left side slightly more generation to force corridor transfers
    for i in range(N):
        x, y = id_xy(i, W)
        if x < W // 2:
            gen_cap_field[i] *= 1.25

    t_idx = np.arange(T)
    load_scale = 1.0 + 0.25 * np.sin(2 * np.pi * (t_idx / 32.0) + 1.0)
    gen_scale = 1.0 + 0.15 * np.sin(2 * np.pi * (t_idx / 32.0) - 0.5)

    # Logs (base model)
    mismatch_log = []
    avg_price_log = []
    p10_log = []
    p90_log = []
    util_log = []
    unmet_log = []
    spill_log = []

    # Heatmaps
    prices_by_step: List[np.ndarray] = []

    # Battery logs
    soc_log = []
    charge_log = []
    spill_before_log = []
    spill_after_log = []

    SoC = 0.0  # MWh

    for tt in range(T):
        load = base_load_field * load_scale[tt]
        gen_cap = gen_cap_field * gen_scale[tt]
        caps = apply_line_limit_shock(W, edges, base_caps, tt, cfg)

        sol = solve_flow_lp(W, H, edges, caps, load, gen_cap, cfg)

        flows = sol["flow"]
        spill = sol["spill"]
        unmet = sol["unmet"]
        prices = sol["prices"]

        prices_by_step.append(prices.copy())

        # Metrics
        mismatch = boundary_mismatch(W, edges, flows, cfg.shock_col)
        avg_price = float(sol["avg_price"][0])
        p10 = float(np.percentile(prices, 10))
        p90 = float(np.percentile(prices, 90))
        util = float(sol["avg_util"][0])

        total_spill_mw = float(spill.sum())
        total_unmet_mw = float(unmet.sum())

        # Battery charges ONLY from real spill
        SoC, charge_mw, spill_remaining_mw = battery_charge_from_spill(SoC, total_spill_mw, cfg)

        # Store logs
        mismatch_log.append(mismatch)
        avg_price_log.append(avg_price)
        p10_log.append(p10)
        p90_log.append(p90)
        util_log.append(util)
        unmet_log.append(total_unmet_mw)
        spill_log.append(total_spill_mw)

        soc_log.append(SoC)
        charge_log.append(charge_mw)
        spill_before_log.append(total_spill_mw)
        spill_after_log.append(spill_remaining_mw)

    # Save heatmaps
    save_price_maps(prices_by_step, W, H, out_prefix="prices")
    print("Saved heatmaps: prices_last.png, prices_mean.png, prices_std.png")

    x = np.arange(T)

    # --- original plots ---
    plt.figure()
    plt.plot(x, mismatch_log, marker="o")
    plt.title("Boundary Mismatch (Flows) – LINE LIMITS + Market")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Mismatch (net flow across corridor)")
    plt.grid(True)

    plt.figure()
    plt.plot(x, avg_price_log, label="Avg Price")
    plt.plot(x, p10_log, label="P10 Price")
    plt.plot(x, p90_log, label="P90 Price")
    plt.title("Prices (Dynamic + Congestion)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, util_log, label="Avg Edge Utilization")
    plt.title("Network Congestion (avg edge utilization)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("Utilization")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, unmet_log, label="Unmet MW")
    plt.plot(x, spill_log, label="Spill MW")
    plt.title("Internal Clearing – Unmet vs Spill")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.grid(True)
    plt.legend()

    # --- battery plots ---
    plt.figure()
    plt.plot(x, soc_log, label="Battery SoC (MWh)")
    plt.title("Battery State of Charge (SoC)")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MWh")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(x, spill_before_log, label="Spill BEFORE battery (MW)")
    plt.plot(x, spill_after_log, label="Spill AFTER battery (MW)")
    plt.plot(x, charge_log, label="Battery charge from spill (MW)")
    plt.title("Battery captures real spill")
    plt.xlabel("Simulation step (block)")
    plt.ylabel("MW")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
